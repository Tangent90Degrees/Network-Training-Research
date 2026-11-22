"""Paper-specific training tricks adapted for ViT.

Methods:
  1. Frozen patch-whitening initialisation  (init_whitening_patch_embed)
  2. Dirac / identity initialisation        (init_dirac_linear)
  3. ScaleBias layer + Lookahead optimizer  (ScaleBias, Lookahead)
  4. Alternating flip                       (handled in data.py)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# 1. Frozen patch-whitening initialisation
#    Adapted from Network.py's init_whitening_conv for a Conv2d patch embed.
# ---------------------------------------------------------------------------

def get_patches(x: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    """Extract all (patch_h × patch_w) patches from a batch of images.
    x: [N, C, H, W]  ->  returns [M, C, patch_h, patch_w]
    """
    return (x.unfold(2, patch_h, patch_h)
              .unfold(3, patch_w, patch_w)
              .permute(0, 2, 3, 1, 4, 5)
              .reshape(-1, x.size(1), patch_h, patch_w)
              .float())


def get_whitening_parameters(patches: torch.Tensor):
    """PCA of flattened patches.
    Returns (eigenvalues, eigenvectors) sorted largest-first.
    eigenvalues:  [D, 1, 1, 1]
    eigenvectors: [D, C, h, w]  (D = C*h*w)
    """
    n, c, h, w = patches.shape
    flat = patches.view(n, -1)                         # [N, D]
    cov  = (flat.T @ flat) / n                         # [D, D]
    eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO='U')
    eigenvalues  = eigenvalues.flip(0).view(-1, 1, 1, 1)
    eigenvectors = eigenvectors.T.reshape(c*h*w, c, h, w).flip(0)
    return eigenvalues, eigenvectors


def init_whitening_patch_embed(patch_embed_layer, x_train: np.ndarray,
                                eps: float = 5e-4, freeze: bool = True):
    """Initialise a PatchEmbed's Conv2d projection with whitened eigenvectors.

    The Conv2d kernel has shape [embed_dim, in_chans, pH, pW].
    We compute eigenvectors of patch covariance (shape [D, C, pH, pW],
    D = C*pH*pW) and fill as many output channels as possible with
    positive/negative eigenvector pairs (like the CNN baseline).
    Extra channels beyond 2*D are left at their default init.

    Args:
        patch_embed_layer: a PatchEmbed instance.
        x_train: float32 np.ndarray [N, 3, 32, 32] in [0,1].
        eps: regularisation for eigenvalue denominator.
        freeze: if True, set requires_grad=False on the conv weight.
    """
    conv = patch_embed_layer.proj
    pH, pW = conv.kernel_size
    x_t = torch.from_numpy(x_train)
    patches = get_patches(x_t, pH, pW)
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    scaled = eigenvectors / torch.sqrt(eigenvalues + eps)   # [D, C, pH, pW]

    D = scaled.shape[0]
    out_channels = conv.weight.shape[0]
    # fill channels with +eig / -eig pairs up to out_channels
    pairs = min(D, out_channels // 2)
    with torch.no_grad():
        conv.weight[:pairs]         = scaled[:pairs]
        conv.weight[pairs:2*pairs]  = -scaled[:pairs]
        # remaining channels (if embed_dim > 2*D): leave as-is

    if freeze:
        conv.weight.requires_grad = False
    print(f"[whitening] initialised {2*pairs}/{out_channels} patch-embed channels, "
          f"frozen={freeze}")


# ---------------------------------------------------------------------------
# 2. Dirac / identity initialisation for linear projection weights
#    The paper uses this for Conv kernels; we adapt it to the linear layers
#    inside attention (Q, K, V projections) and MLP.
# ---------------------------------------------------------------------------

def init_dirac_attention(vit_model):
    """Approximate Dirac init for the QKV and output projections in every
    attention block.  For a square weight matrix W ∈ R^{D×D} this sets
    W = I; for rectangular W ∈ R^{D_out×D_in} it zeroes the matrix and
    places a 1 on every min(D_out, D_in) diagonal entry.
    """
    for blk in vit_model.blocks:
        attn = blk.attn
        # QKV weight: [3*D, D]  — init each third as identity
        D = attn.proj.weight.shape[0]
        with torch.no_grad():
            qkv_w = attn.qkv.weight       # [3D, D]
            qkv_w.zero_()
            for k in range(3):
                sub = qkv_w[k*D:(k+1)*D]  # [D, D]
                sub.copy_(torch.eye(D, device=sub.device))
            if attn.qkv.bias is not None:
                attn.qkv.bias.zero_()
            # Output projection: identity
            attn.proj.weight.copy_(torch.eye(D, device=attn.proj.weight.device))
            if attn.proj.bias is not None:
                attn.proj.bias.zero_()
    print("[dirac] attention QKV and output projections initialised to identity")


# ---------------------------------------------------------------------------
# 3a. ScaleBias — a learned per-feature (scale, bias) wrapper, replacing the
#     standard LayerNorm affine transform.  Applied here as an extra module
#     *after* the ViT head to mirror the Scale(1/9) trick from the CNN.
# ---------------------------------------------------------------------------

class ScaleBias(nn.Module):
    """Learnable per-channel affine transform: y = scale * x + bias.
    Used instead of a raw linear head to let the network learn its own
    output scale (the CNN uses a fixed Scale(1/9) after max-pool).
    """
    def __init__(self, num_features: int, init_scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((num_features,), init_scale))
        self.bias  = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x * self.scale + self.bias


class ViTWithScaleBias(nn.Module):
    """Wraps a ViT and replaces its head with a ScaleBias + linear combo."""
    def __init__(self, vit, embed_dim: int, num_classes: int,
                 init_scale: float = 1.0/9.0):
        super().__init__()
        self.vit = vit
        # Remove original head
        vit.head = nn.Identity()
        self.scale_bias = ScaleBias(embed_dim, init_scale)
        self.head = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, x):
        feats = self.vit.forward_features(x)
        feats = self.scale_bias(feats)
        return self.head(feats)


# ---------------------------------------------------------------------------
# 3b. Lookahead optimizer (Zhang et al., 2019)
# ---------------------------------------------------------------------------

class Lookahead(Optimizer):
    """Lookahead wrapper (k=6, alpha=0.5 as per the original paper)."""

    def __init__(self, base_optimizer: Optimizer, k: int = 6, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        # Slow weights
        self.slow_weights = [
            [p.data.clone() for p in group['params']]
            for group in base_optimizer.param_groups
        ]
        # Expose param_groups so LR schedulers see us
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults

    def state_dict(self):
        return {'step': self._step_count,
                'slow': [[w.cpu() for w in g] for g in self.slow_weights],
                'base': self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self._step_count = state['step']
        self.slow_weights = [[w.to(
            list(self.optimizer.param_groups[gi]['params'])[pi].device)
            for pi, w in enumerate(g)]
            for gi, g in enumerate(state['slow'])]
        self.optimizer.load_state_dict(state['base'])

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step_count += 1
        if self._step_count % self.k == 0:
            for group_slow, group in zip(self.slow_weights,
                                         self.optimizer.param_groups):
                for slow, fast in zip(group_slow, group['params']):
                    slow.add_(self.alpha * (fast.data - slow))
                    fast.data.copy_(slow)
        return loss
