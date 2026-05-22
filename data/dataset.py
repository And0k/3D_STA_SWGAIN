# data/dataset.py
import os, random
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticOceanDataset(Dataset):
    def __init__(self, n_samples, T_window, Z, H, W, channels=2, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        self.n = n_samples
        self.T = T_window
        self.Z = Z
        self.H = H
        self.W = W
        self.C = channels
        base = np.random.randn(n_samples + 50, Z, H, W).astype(np.float32) * 0.1
        for i in range(base.shape[0]):
            z_modes = np.linspace(0, 1, Z)
            for m in range(3):
                amp = (0.5 / (m + 1)) * (1 + 0.5 * np.sin(i * 0.1 + m))
                cx = random.uniform(0, H)
                cy = random.uniform(0, W)
                sigma = random.uniform(3, 8)
                xv, yv = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
                vortex = np.exp(-((xv - cx) ** 2 + (yv - cy) ** 2) / (2 * sigma**2))
                base[i] += amp * (z_modes[:, None, None] * vortex[None, :, :])
        self.base = base
        self.timestamps = np.cumsum(np.random.exponential(scale=3600, size=self.base.shape[0])).astype(
            np.float32
        )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx % (self.base.shape[0] - self.T)
        t_slice = self.base[start : start + self.T]
        u = np.gradient(t_slice, axis=3)
        v = np.gradient(t_slice, axis=2)
        field = np.stack([u, v], axis=1).astype(np.float32)  # (T,C,Z,H,W)
        ts = self.timestamps[start : start + self.T].astype(np.float32)
        mask = np.ones((self.T, 1, self.Z, self.H, self.W), dtype=np.float32)
        sensor_prob = 0.5
        sensor_mask = (np.random.rand(self.Z, self.H, self.W) < sensor_prob).astype(np.float32)
        mask *= sensor_mask[None, :, :, :, :]
        if random.random() < 0.3:
            drop_t = random.randint(0, self.T - 1)
            mask[drop_t] = 0.0
        if random.random() < 0.3:
            t0 = random.randint(0, self.T - 1)
            z0 = random.randint(0, self.Z - 4)
            x0 = random.randint(0, self.H - 8)
            y0 = random.randint(0, self.W - 8)
            mask[t0, :, z0 : z0 + 4, x0 : x0 + 8, y0 : y0 + 8] = 0.0
        field = (field - field.mean()) / (field.std() + 1e-6)
        return {
            "field": torch.from_numpy(field).float(),
            "mask": torch.from_numpy(mask).float(),
            "timestamps": torch.from_numpy(ts).float(),
        }
