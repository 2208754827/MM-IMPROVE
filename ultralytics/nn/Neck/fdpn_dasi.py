"""FDPN-DASI neck block (lightweight migrated implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class DASI(nn.Module):
    """Dynamic Alignment and Scale Integration for multi-level feature fusion."""

    def __init__(self, inc: list[int] | tuple[int, ...], ouc: int):
        super().__init__()
        if len(inc) < 2:
            raise ValueError(f"DASI expects >=2 inputs, got {inc}")
        self.ouc = int(ouc)
        self.align = nn.ModuleList(Conv(int(c), self.ouc, 1, 1) for c in inc)
        self.fuse = Conv(self.ouc * len(inc), self.ouc, 3, 1)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(x) != len(self.align):
            raise ValueError(f"DASI input count mismatch: expected {len(self.align)}, got {len(x)}")

        ref = x[len(x) // 2]
        h, w = ref.shape[2:]
        outs = []
        for xi, layer in zip(x, self.align):
            yi = layer(xi)
            if yi.shape[2:] != (h, w):
                yi = F.interpolate(yi, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(yi)
        return self.fuse(torch.cat(outs, dim=1))

