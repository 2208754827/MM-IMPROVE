"""Wavelet Fusion Unit (WFU) blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class HaarWavelet(nn.Module):
    """Fixed Haar wavelet transform and inverse transform."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)

        weight = torch.ones(4, 1, 2, 2, dtype=torch.float32)
        # Horizontal
        weight[1, 0, 0, 1] = -1
        weight[1, 0, 1, 1] = -1
        # Vertical
        weight[2, 0, 1, 0] = -1
        weight[2, 0, 1, 1] = -1
        # Diagonal
        weight[3, 0, 1, 0] = -1
        weight[3, 0, 0, 1] = -1

        weight = torch.cat([weight] * self.in_channels, dim=0)
        self.register_buffer("haar_weights", weight, persistent=False)

    def forward(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape(x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2)
            out = torch.transpose(out, 1, 2).reshape(x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2)
            return out

        out = x.reshape(x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3])
        out = torch.transpose(out, 1, 2).reshape(x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3])
        return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.in_channels)


class WFU(nn.Module):
    """WFU: fuse a high-resolution feature with a lower-resolution guidance feature."""

    def __init__(self, chn):
        super().__init__()
        if not isinstance(chn, (list, tuple)) or len(chn) != 2:
            raise ValueError(f"WFU expects [dim_big, dim_small], got {chn}")

        dim_big, dim_small = int(chn[0]), int(chn[1])
        self.dim = dim_big
        self.haar_wavelet = HaarWavelet(dim_big)
        self.inverse_haar_wavelet = HaarWavelet(dim_big)

        self.rb = nn.Sequential(
            Conv(dim_big, dim_big, 3),
            nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
        )
        self.channel_transform = nn.Sequential(
            Conv(dim_big + dim_small, dim_big + dim_small, 1),
            nn.Conv2d(dim_big + dim_small, dim_big * 3, kernel_size=1),
        )

    def forward(self, x):
        x_big, x_small = x
        haar = self.haar_wavelet(x_big, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)

        if x_small.shape[-2:] != a.shape[-2:]:
            x_small = F.interpolate(x_small, size=a.shape[-2:], mode="nearest")

        hvd = self.rb(h + v + d)
        a_ = self.channel_transform(torch.cat([x_small, a], dim=1))
        out = self.inverse_haar_wavelet(torch.cat([hvd, a_], dim=1), rev=True)
        return out


__all__ = ["WFU"]

