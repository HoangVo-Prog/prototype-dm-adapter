import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import symmetric_identity_proxy_loss
from .memory import PrototypeMemory


class IdentityProjector(nn.Module):
    def forward(self, features):
        return features.float()


class ResidualIdentityProjector(nn.Module):
    def __init__(self, dim, scale=0.1):
        super().__init__()
        self.scale = float(scale)
        self.residual = nn.Linear(dim, dim)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features):
        features = features.float()
        self.residual.float()
        return features + self.scale * self.residual(features)


def _linear_projector(in_dim, out_dim, orthogonal=False, zero_bias=False):
    projector = nn.Linear(in_dim, out_dim)
    if orthogonal:
        nn.init.orthogonal_(projector.weight)
        nn.init.zeros_(projector.bias)
    elif zero_bias:
        nn.init.zeros_(projector.bias)
    return projector


def _first_device(module, fallback):
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return fallback


def _complete_orthonormal_rows(rows, out_dim, in_dim):
    rows = F.normalize(rows.float(), p=2, dim=1) if rows.numel() > 0 else rows.float()
    completed = [row for row in rows]
    eye = torch.eye(in_dim, dtype=torch.float32, device=rows.device)

    for candidate in eye:
        vector = candidate.clone()
        for row in completed:
            vector = vector - torch.dot(vector, row) * row
        norm = vector.norm()
        if norm > 1e-6:
            completed.append(vector / norm)
        if len(completed) == out_dim:
            break

    if len(completed) < out_dim:
        raise ValueError("could not build enough PCA projector components")
    return torch.stack(completed[:out_dim], dim=0)


def _pca_components(features, out_dim):
    if features.ndim != 2:
        raise ValueError("PCA projector initialization expects a 2D feature tensor")
    in_dim = features.shape[1]
    if out_dim > in_dim:
        raise ValueError(f"prototype_dim ({out_dim}) must be <= feature_dim ({in_dim}) for PCA projector initialization")

    features = features.detach().float().cpu()
    centered = features - features.mean(dim=0, keepdim=True)
    if centered.norm() <= 1e-12:
        return torch.eye(in_dim, dtype=torch.float32)[:out_dim]

    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    rows = vh[: min(out_dim, vh.shape[0])].contiguous()
    if rows.shape[0] < out_dim:
        rows = _complete_orthonormal_rows(rows, out_dim, in_dim)
    return rows[:out_dim]


class PrototypeBranch(nn.Module):
    def __init__(self, args, num_classes, image_dim, text_dim=None):
        super().__init__()
        self.args = args
        self.image_dim = int(image_dim)
        self.text_dim = int(text_dim if text_dim is not None else image_dim)
        self.feature_dim = self.image_dim if self.image_dim == self.text_dim else None
        self.prototype_dim = int(getattr(args, "prototype_dim", 512))
        self.projector_mode = getattr(args, "prototype_projector", "default")
        self._pca_initialized = False

        prototype_feature = getattr(args, "prototype_feature", "auto")
        if prototype_feature not in ("auto", "global"):
            raise ValueError(
                f"prototype-dm-adapter exposes only global prototype features; got --prototype_feature={prototype_feature}"
            )
        self.feature_source = "global"

        self.image_projector, self.text_projector = self._build_projectors()
        self.memory = PrototypeMemory(
            num_classes=num_classes,
            prototypes_per_id=getattr(args, "prototype_per_id", 2),
            dim=self.prototype_dim,
            momentum=getattr(args, "prototype_momentum", 0.2),
        )

    def _require_same_input_dims(self):
        if self.image_dim != self.text_dim:
            raise ValueError(
                f"--prototype_projector {self.projector_mode} requires image_dim == text_dim, "
                f"got image_dim={self.image_dim}, text_dim={self.text_dim}"
            )

    def _require_identity_dims(self):
        self._require_same_input_dims()
        if self.image_dim != self.prototype_dim:
            raise ValueError(
                f"--prototype_projector {self.projector_mode} requires image_dim == text_dim == prototype_dim, "
                f"got image_dim={self.image_dim}, text_dim={self.text_dim}, prototype_dim={self.prototype_dim}"
            )

    def _require_pca_dims(self):
        if self.prototype_dim > self.image_dim or self.prototype_dim > self.text_dim:
            raise ValueError(
                f"--prototype_projector {self.projector_mode} requires prototype_dim <= image_dim/text_dim, "
                f"got prototype_dim={self.prototype_dim}, image_dim={self.image_dim}, text_dim={self.text_dim}"
            )

    def _build_projectors(self):
        mode = self.projector_mode
        if mode == "default":
            return (
                nn.Sequential(nn.Linear(self.image_dim, self.prototype_dim), nn.LayerNorm(self.prototype_dim)),
                nn.Sequential(nn.Linear(self.text_dim, self.prototype_dim), nn.LayerNorm(self.prototype_dim)),
            )
        if mode == "identity":
            self._require_identity_dims()
            return IdentityProjector(), IdentityProjector()
        if mode == "residual_identity":
            self._require_identity_dims()
            scale = getattr(self.args, "prototype_residual_scale", 0.1)
            return ResidualIdentityProjector(self.image_dim, scale), ResidualIdentityProjector(self.text_dim, scale)
        if mode == "random_orthogonal":
            return (
                _linear_projector(self.image_dim, self.prototype_dim, orthogonal=True),
                _linear_projector(self.text_dim, self.prototype_dim, orthogonal=True),
            )
        if mode == "pca_init":
            self._require_pca_dims()
            return (
                _linear_projector(self.image_dim, self.prototype_dim, zero_bias=True),
                _linear_projector(self.text_dim, self.prototype_dim, zero_bias=True),
            )
        if mode == "shared":
            self._require_same_input_dims()
            shared = _linear_projector(self.image_dim, self.prototype_dim)
            return shared, shared
        if mode == "shared_pca_init":
            self._require_same_input_dims()
            self._require_pca_dims()
            shared = _linear_projector(self.image_dim, self.prototype_dim, zero_bias=True)
            return shared, shared
        raise ValueError(f"Unknown prototype projector mode: {mode}")

    def is_ready(self):
        return self.memory.is_ready()

    def needs_pca_init(self):
        return self.projector_mode in ("pca_init", "shared_pca_init") and not self._pca_initialized

    @torch.no_grad()
    def _copy_pca_components(self, projector, features):
        if not isinstance(projector, nn.Linear):
            raise ValueError(f"--prototype_projector {self.projector_mode} expects a linear projector for PCA init")
        projector.float()
        components = _pca_components(features, self.prototype_dim)
        projector.weight.data.copy_(components.to(device=projector.weight.device, dtype=projector.weight.dtype))
        if projector.bias is not None:
            projector.bias.data.zero_()

    @torch.no_grad()
    def initialize_projector_from_features(self, image_features, text_features):
        if not self.needs_pca_init():
            return

        if self.projector_mode == "shared_pca_init":
            features = torch.cat([image_features.detach().cpu(), text_features.detach().cpu()], dim=0)
            self._copy_pca_components(self.image_projector, features)
        else:
            self._copy_pca_components(self.image_projector, image_features)
            self._copy_pca_components(self.text_projector, text_features)
        self._pca_initialized = True

    def _project(self, image_features, text_features):
        self.image_projector.float()
        self.text_projector.float()
        device = _first_device(self, image_features.device)
        image_features = image_features.to(device=device, dtype=torch.float32, non_blocking=True)
        text_features = text_features.to(device=device, dtype=torch.float32, non_blocking=True)
        image_features = F.normalize(self.image_projector(image_features).float(), p=2, dim=1)
        text_features = F.normalize(self.text_projector(text_features).float(), p=2, dim=1)
        return image_features, text_features

    @torch.no_grad()
    def project_for_memory(self, image_features, text_features):
        return self._project(image_features, text_features)

    @torch.no_grad()
    def initialize(self, image_features, text_features, pids):
        if self.needs_pca_init():
            self.initialize_projector_from_features(image_features, text_features)
        image_features, text_features = self._project(image_features, text_features)
        self.initialize_projected(image_features, text_features, pids)

    @torch.no_grad()
    def initialize_projected(self, image_features, text_features, pids):
        prototype_seed = getattr(self.args, "seed", None)
        if prototype_seed is not None:
            prototype_seed = int(prototype_seed) + 1000
        self.memory.initialize(
            F.normalize(image_features.float(), p=2, dim=1).detach(),
            F.normalize(text_features.float(), p=2, dim=1).detach(),
            pids.detach(),
            num_iters=getattr(self.args, "prototype_kmeans_iters", 20),
            seed=prototype_seed,
        )

    def forward(self, image_features, text_features, pids, use_loss_id=True):
        image_features, text_features = self._project(image_features, text_features)
        zero = image_features.sum() * 0.0
        if not self.is_ready():
            return {"proto_id_loss": zero} if use_loss_id else {}

        pids = pids.long()
        ret = {}
        if use_loss_id:
            ret["proto_id_loss"] = symmetric_identity_proxy_loss(
                image_features,
                text_features,
                pids,
                self.memory,
                tau=getattr(self.args, "prototype_tau", 0.05),
                hard_k=getattr(self.args, "prototype_hard_k", 16),
                use_pbt=not getattr(self.args, "no_pbt", False),
            )

        self.memory.ema_update(image_features.detach(), text_features.detach(), pids.detach())
        return ret

    @torch.no_grad()
    def score(self, text_features, image_features):
        was_training = self.training
        self.eval()
        try:
            image_features, text_features = self._project(image_features, text_features)
            scores = self.memory.prototype_score_matrix(text_features, image_features)
        finally:
            self.train(was_training)
        return scores
