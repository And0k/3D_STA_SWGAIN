# Генератор с noise injection, STA и mask‑mixing

# models/generator.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange
from .blocks import Conv3DBlock, Residual3D
from .sta import SpatioTemporalAttention



class LocalFactorizedSTA(nn.Module):
    def __init__(self, dim, heads=4, local_k=7):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim*3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, T, Z, H, W, D = x.shape
        x_flat = x.view(B * T * Z, H * W, D)
        out_spatial = []
        chunk = 16
        for i in range(0, x_flat.shape[0], chunk):
            chunk_x = x_flat[i:i+chunk]
            qkv = self.to_qkv(chunk_x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q / math.sqrt(D / self.heads)
            C_idx = q.shape[0]
            q = q.view(C_idx, -1, self.heads, D // self.heads).permute(0, 2, 1, 3)
            k = k.view(C_idx, -1, self.heads, D // self.heads).permute(0, 2, 3, 1)
            v = v.view(C_idx, -1, self.heads, D // self.heads).permute(0, 2, 1, 3)
            att = torch.matmul(q, k).softmax(dim=-1)
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(C_idx, -1, D)
            out_spatial.append(out)
        out_spatial = torch.cat(out_spatial, dim=0).view(B, T, Z, H, W, D)
        temp_in = out_spatial.permute(0, 2, 3, 4, 1, 5).contiguous().view(B * Z * H * W, T, D)
        qkv = self.to_qkv(temp_in)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q / math.sqrt(D / self.heads)
        N_temp = q.shape[0]
        q = q.view(N_temp, T, self.heads, -1).permute(0, 2, 1, 3)
        k = k.view(N_temp, T, self.heads, -1).permute(0, 2, 3, 1)
        v = v.view(N_temp, T, self.heads, -1).permute(0, 2, 1, 3)
        att = torch.matmul(q, k).softmax(dim=-1)
        out_temp = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(N_temp, T, D)
        out_final = out_temp.view(B, Z, H, W, T, D).permute(0, 4, 1, 2, 3, 5)
        return self.to_out(out_final)

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
        
        
        
        
        
        import logging
        logger = logging.getLogger(__name__)
        B = x_observed.shape[0]
        xc = x_observed.view(B, -1, self.Z, self.H, self.W)
        m = mask.view(B, -1, self.Z, self.H, self.W)
        logger.debug(f"Generator Forward - xc: {xc.shape}, mask: {m.shape}")
        
        if xc.shape[1] != m.shape[1]:
            reps = xc.shape[1] // m.shape[1]
            m_exp = m.repeat(1, reps, 1, 1, 1)
        else:
            m_exp = m
        inp = torch.cat([xc * m_exp, m], dim=1)
        logger.debug(f"Generator input cat shape: {inp.shape}")
        h = self.enc1(inp)
        h = self.enc2(h)
        h = self.res(h)
        # prepare for STA: (B, T, Z, H, W, Cfeat)
        feat_ch = h.shape[1]
        feat = rearrange(h, "b c z h w -> b 1 z h w c", c=feat_ch)  # trick: treat channels as feature dim
        feat = feat.repeat(1, self.T, 1, 1, 1, 1)
        feat = self.sta(feat)
        feat = rearrange(feat, "b t z h w c -> b c z h w", c=feat_ch)
        h = feat
        out = self.dec1(h)
        out = self.out_conv(out)
        out = out.view(B, T, self.C, self.Z, self.H, self.W)
        return out
