# Spatio‑Temporal Attention (factorized simple version)

import torch
import torch.nn as nn
from einops import rearrange


class SpatioTemporalAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.proj = nn.Linear(inner, dim)

    def forward(self, x):
        # x: (B, T, Z, H, W, C)
        B, T, Z, H, W, C = x.shape
        n = T * Z * H * W
        x_flat = x.view(B, n, C)
        q = self.to_q(x_flat)
        k = self.to_k(x_flat)
        v = self.to_v(x_flat)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (q.shape[-1] ** 0.5))
        att = att.softmax(dim=-1)
        out = att @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)
        return out.view(B, T, Z, H, W, C)
