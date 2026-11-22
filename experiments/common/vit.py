"""Vision Transformer (ViT) for CIFAR-10 (32×32).

The patch embedding is kept separate so the frozen patch-whitening
initialisation (see paper §3.1) can be applied to it.  The rest of the
transformer follows a standard ViT design.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Conv-based patch embedding.  Exposes `.proj.weight` so whitening /
    Dirac initialisation can be applied directly, mirroring the CNN code."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size,
                              bias=True)

    def forward(self, x):
        x = self.proj(x)            # [B, E, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        return x


def drop_path(x: torch.Tensor, drop_prob: float, training: bool):
    """Per-sample stochastic depth (DropPath). Drops entire residual
    paths during training, regularising deep ViTs."""
    if drop_prob <= 0. or not training:
        return x
    keep = 1. - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rand = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    rand.floor_()
    return x.div(keep) * rand


class DropPath(nn.Module):
    """Module wrapper for drop_path; drops the whole path per-sample."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1,2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path_rate=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn  = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        # DropPath applied to the attention residual path
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. \
                          else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp   = MLP(dim, mlp_hidden, dim, drop)
        # DropPath applied to the MLP residual path
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. \
                          else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    """A small ViT sized for CIFAR-10.

    Defaults: patch=4, dim=192, depth=9, heads=4  (~5.7M params) — chosen so a
    full run fits comfortably in a few minutes on an RTX 4090 while still
    being a genuine transformer baseline.

    drop_path_rate is linearly scaled across blocks (0 → max) following DeiT.
    """

    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=192, depth=9, num_heads=4, mlp_ratio=4.0,
                 drop_rate=0.1, attn_drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop   = nn.Dropout(drop_rate)

        # Linearly scale drop_path across depth: block 0 → 0, last block → max
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path_rate=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # Plain linear head; ScaleBias wrapper added externally when needed
        self.head = nn.Linear(embed_dim, num_classes, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # Standard He init fallback; specialised inits override later
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)                       # [B, N, E]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1)                # [B, N+1, E]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        feats = self.forward_features(x)
        return self.head(feats)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
