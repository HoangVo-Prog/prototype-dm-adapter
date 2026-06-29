from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1)  # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader  # gallery
        self.txt_loader = txt_loader  # query
        self.logger = logging.getLogger("dm-adapter.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        # print(model)
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption, l_aux=0).cpu()
            qids.append(pid.view(-1))  # flatten
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img, l_aux=0).cpu()
            gids.append(pid.view(-1))  # flatten
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids

    def eval(self, model, i2t_metric=False, return_metrics=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
        gfeats = F.normalize(gfeats, p=2, dim=1)  # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        t2i_rsum = t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_rsum])
        top1 = float(t2i_cmc[0])
        metrics = {
            "t2i/R1": float(t2i_cmc[0]),
            "t2i/R5": float(t2i_cmc[4]),
            "t2i/R10": float(t2i_cmc[9]),
            "t2i/mAP": float(t2i_mAP),
            "t2i/mINP": float(t2i_mINP),
            "t2i/rSum": float(t2i_rsum),
        }
        best_metrics = {
            "task": "t2i",
            "R1": float(t2i_cmc[0]),
            "R5": float(t2i_cmc[4]),
            "R10": float(t2i_cmc[9]),
            "mAP": float(t2i_mAP),
            "mINP": float(t2i_mINP),
            "rSum": float(t2i_rsum),
        }

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10,
                                                 get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            i2t_rsum = i2t_cmc[0] + i2t_cmc[4] + i2t_cmc[9]
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP, i2t_rsum])
            metrics.update({
                "i2t/R1": float(i2t_cmc[0]),
                "i2t/R5": float(i2t_cmc[4]),
                "i2t/R10": float(i2t_cmc[9]),
                "i2t/mAP": float(i2t_mAP),
                "i2t/mINP": float(i2t_mINP),
                "i2t/rSum": float(i2t_rsum),
            })
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["rSum"] = lambda f, v: f"{v:.2f}"
        self.logger.info('\n' + str(table))
        self.logger.info('\n' + "best R1 = " + str(top1))

        if return_metrics:
            return top1, metrics, best_metrics
        return top1
