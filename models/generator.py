# Генератор с noise injection, STA и mask‑mixing

# models/generator.py
import torch
import torch.nn as nn
from einops import rearrange
from .blocks import Conv3DBlock, Residual3D
from .sta import SpatioTemporalAttention


class Generator(nn.Module):
    def __init__(self, T_window, Z, H, W, channels=2, base_dim=32, time_dim=16):
        super().__init__()
        self.T = T_window
        self.Z = Z
        self.H = H
        self.W = W
        self.C = channels
        in_ch = (channels + 1) * T_window
        self.enc1 = Conv3DBlock(in_ch, base_dim)
        self.enc2 = Conv3DBlock(base_dim, base_dim * 2)
        self.res = Residual3D(base_dim * 2)
        self.sta = SpatioTemporalAttention(dim=base_dim * 2, heads=4, dim_head=16)
        self.dec1 = Conv3DBlock(base_dim * 2, base_dim)
        self.out_conv = nn.Conv3d(base_dim, channels * T_window, 3, padding=1)

    def forward(self, x_observed, mask, noise=None):
        # x_observed: (B, T, C, Z, H, W)
        B = x_observed.shape[0]
        T = self.T
        xc = x_observed.view(B, -1, self.Z, self.H, self.W)
        m = mask.view(B, -1, self.Z, self.H, self.W)
        inp = torch.cat([xc * m, m], dim=1)
        h = self.enc1(inp)
        h = self.enc2(h)
        h = self.res(h)
        # prepare for STA: (B, T, Z, H, W, Cfeat)
        feat_ch = h.shape[1]
        feat = rearrange(h, "b c z h w -> b 1 z h w c", c=feat_ch)  # trick: treat channels as feature dim
        feat = feat.repeat(1, T, 1, 1, 1, 1)
        feat = self.sta(feat)
        feat = rearrange(feat, "b t z h w c -> b c z h w", c=feat_ch)
        h = feat
        out = self.dec1(h)
        out = self.out_conv(out)
        out = out.view(B, T, self.C, self.Z, self.H, self.W)
        return out
