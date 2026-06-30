from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import math
from .prototype import PrototypeBranch
    
class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg, state_dict = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size, args.num_experts, args.topk, args.reduction)

        self.embed_dim = base_cfg['embed_dim']
        self.prototype_enabled = (
            getattr(args, "prototype", False)
            or getattr(args, "use_loss_id", False)
        )

        # new add vs V5
        self.apply(self.init_weights) # random init must before loading pretrain
        self.base_model.load_param(state_dict)

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
            
        if 'id' in args.loss_names or 'imkt' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
        
        for i in range(12):
            for j in range(args.num_experts):
                nn.init.kaiming_uniform_(self.base_model.visual.transformer.resblocks[i].feed_forward.experts[j].down.weight, a=math.sqrt(5))
                nn.init.zeros_(self.base_model.visual.transformer.resblocks[i].feed_forward.experts[j].down.bias)
                nn.init.zeros_(self.base_model.visual.transformer.resblocks[i].feed_forward.experts[j].up.weight)
                nn.init.zeros_(self.base_model.visual.transformer.resblocks[i].feed_forward.experts[j].up.bias)


        for i in range(12):
            for j in range(args.num_experts):
                nn.init.kaiming_uniform_(self.base_model.transformer.resblocks[i].feed_forward.experts[j].down.weight, a=math.sqrt(5))
                nn.init.zeros_(self.base_model.transformer.resblocks[i].feed_forward.experts[j].down.bias)
                nn.init.zeros_(self.base_model.transformer.resblocks[i].feed_forward.experts[j].up.weight)
                nn.init.zeros_(self.base_model.transformer.resblocks[i].feed_forward.experts[j].up.bias)

        if self.prototype_enabled:
            self.prototype_branch = PrototypeBranch(
                args=args,
                num_classes=num_classes,
                image_dim=self.embed_dim,
                text_dim=self.embed_dim,
            )
        else:
            self.prototype_branch = None
                
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()              

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+') if l.strip() and l.strip() != 'proto']
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image, l_aux):
        outputs = self.base_model.encode_image(image, l_aux)
        x = outputs[0]
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text, l_aux):
        outputs = self.base_model.encode_text(text, l_aux)
        x = outputs[0]
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def _compute_host_embeddings(self, images, caption_ids):
        image_feats, text_feats, l_aux = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        return {
            "image_tokens": image_feats,
            "text_tokens": text_feats,
            "i_feats": i_feats,
            "t_feats": t_feats,
            "l_aux": l_aux,
        }

    def select_prototype_features(self, outputs, batch):
        return outputs["i_feats"], outputs["t_feats"]

    @torch.no_grad()
    def extract_prototype_features(self, batch):
        outputs = self._compute_host_embeddings(batch['images'], batch['caption_ids'])
        return self.select_prototype_features(outputs, batch)

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        outputs = self._compute_host_embeddings(images, caption_ids)
        image_feats = outputs["image_tokens"]
        i_feats = outputs["i_feats"]
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        text_feats = outputs["text_tokens"]
        t_feats = outputs["t_feats"]
        l_aux = outputs["l_aux"]

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'aux' in self.current_task:
            #print(f'l_aux:{l_aux}')
            ret.update({'aux_loss': 0.5 * l_aux})

        if 'triplet_enhance' in self.current_task:
            ret.update({'triplet_enhance_loss': 0.5 * objectives.compute_triplet_enhance(i_feats, t_feats, batch['pids'])})

        if 'triplet_enhance_shuffle' in self.current_task:
            ret.update({'triplet_enhance_shuffle_loss': 0.5 * objectives.compute_triplet_enhance_shuffle(i_feats, t_feats, batch['pids'])})

        if 'triplet' in self.current_task:
            ret.update({'triplet_loss':0.5 * objectives.compute_triplet(i_feats, t_feats)})

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'imkt' in self.current_task:
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'imkt_loss': objectives.compute_imkt(text_logits, batch['pids'])})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)

            mlm_labels = batch['mlm_labels'].reshape(-1)


            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if self.prototype_enabled and self.prototype_branch is not None:
            proto_image_feats, proto_text_feats = self.select_prototype_features(outputs, batch)
            ret["_diag"] = {
                "host_image_feats": i_feats.detach(),
                "host_text_feats": t_feats.detach(),
                "proto_image_feats": proto_image_feats.detach(),
                "proto_text_feats": proto_text_feats.detach(),
                "pids": batch["pids"].detach(),
                "indices": batch.get("index", None),
            }
            proto_ret = self.prototype_branch(
                proto_image_feats,
                proto_text_feats,
                batch['pids'],
                use_loss_id=getattr(self.args, "use_loss_id", False),
            )
            if "proto_id_loss" in proto_ret:
                ret["proto_id_loss"] = proto_ret["proto_id_loss"] * getattr(self.args, "prototype_id_weight", 0.2)

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # Keep the original fp16 conversion path; deterministic mode controls CUDA kernels,
    # but near-tied fp16 decisions may still differ across GPU architectures.
    convert_weights(model)
    return model
