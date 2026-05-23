
import torch
import torch.nn as nn
import logging
import os
import psutil

logger = logging.getLogger(__name__)

def log_mem(msg):
    """Logs current process Resident Set Size (RSS) in MB for memory monitoring."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    logging.debug(f'[CRITIC MEM] {msg} | Current RSS: {mem:.2f} MB')

class Conv3DBlock(nn.Module):
    """Standard 3D Convolutional block: Conv3d -> BatchNorm3d -> LeakyReLU."""
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel, stride, padding),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    """3D Critic that merges Time and Depth dimensions for Conv3d compatibility (6D -> 5D)."""
    def __init__(self, base_dim=16, T_window=4, Z=8, H=32, W=32):
        super().__init__()
        self.base_dim = base_dim
        # Input: 4 channels (field) + 4 channels (mask) = 8 channels total
        self.conv1 = Conv3DBlock(8, base_dim)
        self.conv2 = Conv3DBlock(base_dim, base_dim * 2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(base_dim * 2, 1)
        logging.info(f'Critic initialized: base_dim={base_dim}, head_in={base_dim*2}')

    def forward(self, x, mask):
        # Initial 6D shape: [B, C, T, Z, H, W]
        log_mem(f'Forward Start | x: {list(x.shape)}, mask: {list(mask.shape)}')

        # 1. Handle temporal dimension broadcasting for 6D tensors
        if x.shape[2] != mask.shape[2]:
            logging.debug(f'Expanding mask T={mask.shape[2]} to match x T={x.shape[2]}')
            mask = mask.expand(-1, -1, x.shape[2], -1, -1, -1)

        # 2. Concatenate along channel dimension (dim 1) -> [B, 8, T, Z, H, W]
        combined = torch.cat([x, mask], dim=1)

        # 3. Reshape 6D to 5D for Conv3d: [B, C, T, Z, H, W] -> [B, C, T*Z, H, W]
        # This treats temporal and vertical dimensions as a single spatial-temporal volume.
        B, C, T, Z, H, W = combined.shape
        combined_5d = combined.view(B, C, T * Z, H, W)

        # 4. Forward through 3D convolutional layers
        x = self.conv1(combined_5d)
        x = self.conv2(x)
        x = self.pool(x)

        # 5. Final scoring head
        x = x.view(x.size(0), -1)
        out = self.head(x)

        log_mem("Forward pass completed")
        return out
