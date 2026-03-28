"""PAC-APN neck blocks (lightweight migrated implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, ConvTranspose, autopad


class ParallelAtrousConv(nn.Module):
    """Parallel atrous convolutions for context aggregation."""

    def __init__(self, inc: int, ratio=(1, 2, 3)):
        super().__init__()
        c_mid = max(int(inc // 2), 1)
        self.conv1 = Conv(inc, inc, k=3, d=ratio[0])
        self.conv2 = Conv(inc, c_mid, k=3, d=ratio[1])
        self.conv3 = Conv(inc, c_mid, k=3, d=ratio[2])
        self.conv4 = Conv(inc * 2, inc, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv4(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1))


class CSP_PAC(nn.Module):
    """CSP bottleneck with ParallelAtrousConv."""

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        c_ = max(int(c2 * e), 1)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = ParallelAtrousConv(c_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class AttentionUpsample(nn.Module):
    """Channel-gated upsample with two parallel upsample branches."""

    def __init__(self, inc: int):
        super().__init__()
        c_half = max(inc // 2, 1)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate = nn.Sequential(nn.Conv2d(inc, inc, 1), nn.Hardsigmoid())
        self.conv = Conv(inc, inc, k=1)
        self.up_branch1 = ConvTranspose(inc, c_half, 2, 2)
        self.up_branch2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), Conv(inc, c_half, k=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_gate = self.gate(self.globalpool(x))
        x_up = torch.cat([self.up_branch1(x), self.up_branch2(x)], dim=1) * channel_gate
        return self.conv(x_up)


class AttentionDownsample(nn.Module):
    """Channel-gated downsample with strided-conv and pooled branches."""

    def __init__(self, inc: int):
        super().__init__()
        c_half = max(inc // 2, 1)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate = nn.Sequential(nn.Conv2d(inc, inc, 1), nn.Hardsigmoid())
        self.conv = Conv(inc, inc, k=1)
        self.down_branch1 = Conv(inc, c_half, 3, 2)
        self.down_branch2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), Conv(inc, c_half, k=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_gate = self.gate(self.globalpool(x))
        x_down = torch.cat([self.down_branch1(x), self.down_branch2(x)], dim=1) * channel_gate
        return self.conv(x_down)

