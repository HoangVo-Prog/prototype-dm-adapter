import torch
import torch.nn.functional as F


def _masked_logsumexp(values, mask, dim):
    return values.masked_fill(~mask, float("-inf")).logsumexp(dim=dim)


def identity_proxy_contrastive(features, pids, prototypes, proto_pids, tau=0.05, hard_k=16):
    features = F.normalize(features.float(), p=2, dim=1)
    prototypes = F.normalize(prototypes.to(features.device).float(), p=2, dim=1)
    proto_pids = proto_pids.to(features.device).long()
    pids = pids.to(features.device).long()

    logits = features @ prototypes.t()
    scaled = logits / max(float(tau), 1e-6)
    pos_mask = proto_pids.unsqueeze(0).eq(pids.unsqueeze(1))
    neg_mask = ~pos_mask
    if not neg_mask.any():
        return features.new_zeros(())

    pos_lse = _masked_logsumexp(scaled, pos_mask, dim=1)
    neg_logits = scaled.masked_fill(~neg_mask, float("-inf"))
    k = min(max(int(hard_k), 1), neg_logits.shape[1])
    hard_values, hard_idx = neg_logits.topk(k=k, dim=1)
    hard_mask = torch.zeros_like(neg_mask)
    hard_mask.scatter_(1, hard_idx, torch.isfinite(hard_values))
    neg_lse = _masked_logsumexp(scaled, hard_mask, dim=1)

    valid = torch.isfinite(pos_lse) & torch.isfinite(neg_lse)
    if not valid.any():
        return features.new_zeros(())
    return F.softplus(neg_lse[valid] - pos_lse[valid]).mean()


def identity_proxy_banks(memory, use_pbt=True):
    if use_pbt:
        return memory.text_to_image, memory.image_to_text
    return memory.text_prototypes, memory.image_prototypes


def symmetric_identity_proxy_loss(image_features, text_features, pids, memory, tau=0.05, hard_k=16, use_pbt=True):
    image_prototypes, text_prototypes = identity_proxy_banks(memory, use_pbt=use_pbt)
    image_loss = identity_proxy_contrastive(
        image_features, pids, image_prototypes, memory.proto_pids, tau=tau, hard_k=hard_k
    )
    text_loss = identity_proxy_contrastive(
        text_features, pids, text_prototypes, memory.proto_pids, tau=tau, hard_k=hard_k
    )
    return 0.5 * (image_loss + text_loss)
