# Generator with noise injection, STA and mask-mixing
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import logging
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from .blocks import Conv3DBlock, Residual3D
from .sta import SpatioTemporalAttention

logger = logging.getLogger(__name__)

class Generator(nn.Module):
    def __init__(self, T_window, Z, H, W, channels=2, base_dim=32, time_dim=16):
        super().__init__()
        self.T = T_window
        self.Z = Z
        self.H = H
        self.W = W
        self.C = channels

        in_ch = (channels + 1) * T_window
        # Based on logs, the actual output is 16 channels. 
        # We will explicitly define dimensions to avoid ambiguity.
        self.enc1 = Conv3DBlock(in_ch, 16)
        self.enc2 = Conv3DBlock(16, 16)
        self.res = Residual3D(16)

        # STA dim now matches the 16 channels from enc2/res
        self.sta = SpatioTemporalAttention(dim=16, heads=4, dim_head=4)

        self.dec1 = Conv3DBlock(16, 16)
        self.out_conv = nn.Conv3d(16, channels * T_window, 3, padding=1)

    def forward(self, x_observed, mask, noise=None):
        B = x_observed.shape[0]
        x_in = torch.cat([x_observed.view(B, -1, self.Z, self.H, self.W),
                          mask.view(B, -1, self.Z, self.H, self.W)], dim=1)

        x = checkpoint(self.enc1, x_in, use_reentrant=False)
        x = checkpoint(self.enc2, x, use_reentrant=False)
        x = checkpoint(self.res, x, use_reentrant=False)

        # Reshape for STA: (B, T, Z, H, W, C)
        x_for_sta = x.permute(0, 2, 3, 4, 1).unsqueeze(1).contiguous()

        x_sta = checkpoint(self.sta, x_for_sta, use_reentrant=False)

        # Restore to (B, C, Z, H, W)
        x = x_sta.squeeze(1).permute(0, 4, 1, 2, 3).contiguous()

        x = checkpoint(self.dec1, x, use_reentrant=False)
        out = self.out_conv(x)

        return out.view(B, self.T, self.C, self.Z, self.H, self.W)
