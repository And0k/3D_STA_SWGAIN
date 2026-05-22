# WGAN‑GP Critic с простыми 3D‑блоками и STA‑поддержкой

# models/critic.py
import torch
import torch.nn as nn
from .blocks import Conv3DBlock
from .sta import SpatioTemporalAttention
from einops import rearrange


class Critic(nn.Module):
    def __init__(self, T_window, Z, H, W, channels=2, base_dim=32):
        super().__init__()
        in_ch = (channels + 1) * T_window
        self.conv1 = Conv3DBlock(in_ch, base_dim)
        self.conv2 = Conv3DBlock(base_dim, base_dim * 2)
        self.sta = SpatioTemporalAttention(dim=base_dim * 2, heads=4, dim_head=16)
        self.head = nn.Linear(base_dim * 2 * Z * H * W, 1)

    def forward(self, x, mask):
        # x: (B, T, C, Z, H, W)
        B = x.shape[0]
        T = x.shape[1]
        xc = x.view(B, -1, x.shape[3], x.shape[4], x.shape[5])
        m = mask.view(B, -1, x.shape[3], x.shape[4], x.shape[5])
        inp = torch.cat([xc * m, m], dim=1)
        h = self.conv1(inp)
        h = self.conv2(h)
        # STA expects (B,T,Z,H,W,C)
        feat_ch = h.shape[1]
        feat = rearrange(h, "b c z h w -> b 1 z h w c", c=feat_ch)
        feat = feat.repeat(1, T, 1, 1, 1, 1)
        feat = self.sta(feat)
        feat = rearrange(feat, "b t z h w c -> b c z h w", c=feat_ch)
        h = feat
        return self.head(h.view(B, -1))
