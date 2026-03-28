"""FDPN neck blocks (lightweight migrated implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import ADown
from ultralytics.nn.modules.conv import Conv, autopad


class FocusFeature(nn.Module):
    """Focus Diffusion Pyramid Network feature aggregation."""

    def __init__(self, inc: list[int] | tuple[int, ...], kernel_sizes=(5, 7, 9, 11), e: float = 0.5):
        super().__init__()
        if len(inc) < 3:
            raise ValueError(f"FocusFeature expects >=3 inputs, got {inc}")

        hidc = max(int(inc[1] * e), 1)
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv(int(inc[0]), hidc, 1),
        )
        self.conv2 = Conv(int(inc[1]), hidc, 1) if e != 1 else nn.Identity()
        self.conv3 = ADown(int(inc[2]), hidc)

        self.dw_conv = nn.ModuleList(
            nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3)
            for k in kernel_sizes
        )
        self.pw_conv = Conv(hidc * 3, hidc * 3)

    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        feature = self.pw_conv(feature)
        return x + feature

