# Небольшие 3D‑блоки и residual для генератора/критика.

import torch.nn as nn

class Conv3DBlock(nn.Module):

    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, k, stride=stride, padding=padding),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Residual3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = Conv3DBlock(ch, ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.norm(h)
        return nn.ReLU()(x + h)
