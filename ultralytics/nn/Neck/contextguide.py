"""ContextGuideFPN blocks migrated from RTDETR-main (minimal dependency version)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class SEAttention(nn.Module):
    """Squeeze-and-excitation attention used by ContextGuideFusionModule."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.avg_pool(x))


class ContextGuideFusionModule(nn.Module):
    """Context-guided bidirectional feature fusion."""

    def __init__(self, inc: list[int] | tuple[int, int]):
        super().__init__()
        if len(inc) != 2:
            raise ValueError(f"ContextGuideFusionModule expects 2 input channels, got {inc}")
        c0, c1 = int(inc[0]), int(inc[1])
        self.adjust_conv = nn.Identity() if c0 == c1 else Conv(c0, c1, k=1)
        self.se = SEAttention(c1 * 2)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        x_cat = self.se(torch.cat([x0, x1], dim=1))
        x0_w, x1_w = torch.split(x_cat, [x0.shape[1], x1.shape[1]], dim=1)
        x0_w = x0 * x0_w
        x1_w = x1 * x1_w
        return torch.cat([x0 + x1_w, x1 + x0_w], dim=1)
