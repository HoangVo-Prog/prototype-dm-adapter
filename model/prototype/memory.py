import torch
import torch.nn as nn
import torch.nn.functional as F

from .kmeans import identity_kmeans


class PrototypeMemory(nn.Module):
    def __init__(self, num_classes, prototypes_per_id, dim, momentum=0.2):
        super().__init__()
        self.num_classes = int(num_classes)
        self.prototypes_per_id = int(prototypes_per_id)
        self.dim = int(dim)
        self.momentum = float(momentum)
        total = self.num_classes * self.prototypes_per_id

        self.register_buffer("image_prototypes", torch.zeros(total, self.dim))
        self.register_buffer("text_prototypes", torch.zeros(total, self.dim))
        self.register_buffer("text_to_image", torch.zeros(total, self.dim))
        self.register_buffer("image_to_text", torch.zeros(total, self.dim))
        self.register_buffer(
            "proto_pids",
            torch.arange(self.num_classes).repeat_interleave(self.prototypes_per_id).long(),
        )
        self.register_buffer("initialized", torch.tensor(False))

    @property
    def total_prototypes(self):
        return self.num_classes * self.prototypes_per_id

    def is_ready(self):
        return bool(self.initialized.item())

    def _validate_pids(self, pids):
        if pids.numel() == 0:
            raise ValueError("prototype initialization received no samples")
        if pids.min().item() < 0 or pids.max().item() >= self.num_classes:
            raise ValueError(
                f"prototype pids must be in [0, {self.num_classes - 1}], "
                f"got min={pids.min().item()} max={pids.max().item()}"
            )
        expected = torch.arange(self.num_classes, dtype=torch.long)
        unique = torch.unique(pids.detach().cpu().long(), sorted=True)
        if unique.numel() != expected.numel() or not torch.equal(unique, expected):
            missing = sorted(set(expected.tolist()) - set(unique.tolist()))
            preview = missing[:10]
            raise ValueError(
                "prototype initialization did not cover every train identity; "
                f"missing {len(missing)} identities, first missing={preview}"
            )

    @torch.no_grad()
    def initialize(self, image_features, text_features, pids, num_iters=20, seed=None):
        image_features = F.normalize(image_features.float(), p=2, dim=1)
        text_features = F.normalize(text_features.float(), p=2, dim=1)
        pids = pids.long()
        self._validate_pids(pids)

        image_bank = identity_kmeans(
            image_features,
            pids,
            self.num_classes,
            self.prototypes_per_id,
            num_iters=num_iters,
            seed=seed,
        )
        text_bank = identity_kmeans(
            text_features,
            pids,
            self.num_classes,
            self.prototypes_per_id,
            num_iters=num_iters,
            seed=None if seed is None else int(seed) + 1,
        )

        self.image_prototypes.copy_(image_bank.to(self.image_prototypes.device))
        self.text_prototypes.copy_(text_bank.to(self.text_prototypes.device))
        self._rebuild_pbt(image_features, text_features, pids)
        self.initialized.fill_(True)

    def assign_identity(self, features, pids, bank):
        features = F.normalize(features.float(), p=2, dim=1)
        pids = pids.long().to(features.device)
        bank = bank.to(features.device)
        local_bank = bank.view(self.num_classes, self.prototypes_per_id, self.dim)[pids]
        sims = torch.bmm(local_bank, features.unsqueeze(-1)).squeeze(-1)
        local_idx = sims.argmax(dim=1)
        return pids * self.prototypes_per_id + local_idx

    @torch.no_grad()
    def _group_means(self, assignments, features, bank):
        assignments = assignments.to(bank.device).long()
        features = features.to(device=bank.device, dtype=bank.dtype)
        sums = torch.zeros_like(bank)
        counts = torch.zeros(bank.shape[0], 1, device=bank.device, dtype=bank.dtype)
        sums.index_add_(0, assignments, features)
        counts.index_add_(
            0,
            assignments,
            torch.ones(assignments.shape[0], 1, device=bank.device, dtype=bank.dtype),
        )
        valid_mask = counts.squeeze(1) > 0
        valid = valid_mask.nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            return valid, bank.new_empty((0, bank.shape[1]))
        means = sums[valid] / counts[valid].clamp_min(1.0)
        return valid, F.normalize(means, p=2, dim=1)

    @torch.no_grad()
    def _mean_scatter(self, bank, assignments, features):
        valid, means = self._group_means(assignments, features, bank)
        if valid.numel() > 0:
            bank[valid] = means.to(device=bank.device, dtype=bank.dtype)

    @torch.no_grad()
    def _ema_scatter(self, bank, assignments, features):
        valid, means = self._group_means(assignments, features, bank)
        if valid.numel() > 0:
            means = means.to(device=bank.device, dtype=bank.dtype)
            bank[valid] = F.normalize(
                (1.0 - self.momentum) * bank[valid] + self.momentum * means,
                p=2,
                dim=1,
            )

    @torch.no_grad()
    def _rebuild_pbt(self, image_features, text_features, pids):
        image_features = F.normalize(image_features.float(), p=2, dim=1).to(self.image_prototypes.device)
        text_features = F.normalize(text_features.float(), p=2, dim=1).to(self.text_prototypes.device)
        pids = pids.long().to(self.proto_pids.device)

        self.text_to_image.copy_(self.image_prototypes)
        self.image_to_text.copy_(self.text_prototypes)

        image_assign = self.assign_identity(image_features, pids, self.image_prototypes)
        text_assign = self.assign_identity(text_features, pids, self.text_prototypes)
        self._mean_scatter(self.text_to_image, text_assign, image_features)
        self._mean_scatter(self.image_to_text, image_assign, text_features)

    @torch.no_grad()
    def ema_update(self, image_features, text_features, pids):
        if not self.is_ready():
            return

        image_features = F.normalize(image_features.float(), p=2, dim=1)
        text_features = F.normalize(text_features.float(), p=2, dim=1)
        pids = pids.long()

        image_assign = self.assign_identity(image_features, pids, self.image_prototypes)
        text_assign = self.assign_identity(text_features, pids, self.text_prototypes)
        self._ema_scatter(self.image_prototypes, image_assign, image_features)
        self._ema_scatter(self.text_prototypes, text_assign, text_features)
        self._ema_scatter(self.text_to_image, text_assign, image_features)
        self._ema_scatter(self.image_to_text, image_assign, text_features)

    def assign_global(self, features, bank, chunk_size=4096):
        features = F.normalize(features.float(), p=2, dim=1)
        bank = F.normalize(bank.to(features.device).float(), p=2, dim=1)
        assignments = []
        for start in range(0, features.shape[0], max(int(chunk_size), 1)):
            sims = features[start:start + chunk_size] @ bank.t()
            assignments.append(sims.argmax(dim=1))
        return torch.cat(assignments, dim=0)

    @torch.no_grad()
    def prototype_score_matrix(self, text_features, image_features):
        if not self.is_ready():
            return text_features.new_zeros((text_features.shape[0], image_features.shape[0]))

        text_features = F.normalize(text_features.float(), p=2, dim=1)
        image_features = F.normalize(image_features.float(), p=2, dim=1)
        text_idx = self.assign_global(text_features, self.text_prototypes)
        image_idx = self.assign_global(image_features, self.image_prototypes)

        visual_pbt = self.text_to_image.to(image_features.device).float()[text_idx]
        visual_cluster = self.image_prototypes.to(image_features.device).float()[image_idx]
        text_pbt = self.image_to_text.to(text_features.device).float()[image_idx]
        text_cluster = self.text_prototypes.to(text_features.device).float()[text_idx]

        visual_scores = F.normalize(visual_pbt, p=2, dim=1) @ F.normalize(visual_cluster, p=2, dim=1).t()
        text_scores = F.normalize(text_cluster, p=2, dim=1) @ F.normalize(text_pbt, p=2, dim=1).t()
        return visual_scores + text_scores
