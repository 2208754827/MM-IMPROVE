"""HyperCompute neck blocks (minimal port for RTDETRMM variants)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import Conv, DWConv


class MessageAgg(nn.Module):
    """Aggregate node/edge messages with mean normalization."""

    def forward(self, x: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(path, x)
        norm = 1.0 / torch.sum(path, dim=2, keepdim=True)
        norm[torch.isinf(norm)] = 0
        return norm * x


class HyPConv(nn.Module):
    """A lightweight hypergraph conv: v->e and e->v message passing."""

    def __init__(self, c: int):
        super().__init__()
        self.fc = nn.Linear(c, c)
        self.v2e = MessageAgg()
        self.e2v = MessageAgg()

    def forward(self, x: torch.Tensor, h_mat: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        e = self.v2e(x, h_mat.transpose(1, 2).contiguous())
        x = self.e2v(e, h_mat)
        return x


class HyperComputeModule(nn.Module):
    """
    Hypergraph computation block.

    Args:
        c (int): input/output channels.
        threshold (float): distance threshold used to build hypergraph adjacency.
    """

    def __init__(self, c: int = 256, threshold: float = 10.0):
        super().__init__()
        self.threshold = float(threshold)
        self.hgconv = HyPConv(c)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2).contiguous()  # [B, N, C]
        feat = x.clone()
        dist = torch.cdist(feat, feat)
        h_mat = (dist < self.threshold).to(dtype=x.dtype)
        x = self.hgconv(x, h_mat) + x
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        return self.act(self.bn(x))


class MANet(nn.Module):
    """
    MANet block.

    YAML args style used here: [c2, shortcut, p, kernel_size]
    """

    def __init__(
        self,
        c1: int = 256,
        c2: int = 256,
        n: int = 1,
        shortcut: bool = True,
        p: float = 1.0,
        kernel_size: int = 3,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        hidden = int(c2 * e)
        self.c = hidden
        self.cv_first = Conv(c1, 2 * hidden, 1, 1)
        self.cv_final = Conv((4 + n) * hidden, c2, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(hidden, hidden, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )
        self.cv_block_1 = Conv(2 * hidden, hidden, 1, 1)
        dim_hid = int(p * 2 * hidden)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * hidden, dim_hid, 1, 1),
            DWConv(dim_hid, dim_hid, kernel_size, 1),
            Conv(dim_hid, hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        out = [y0, y1, y2, y3]
        out.extend(m(out[-1]) for m in self.m)
        return self.cv_final(torch.cat(out, 1))


class StarBlock(nn.Module):
    """Star operation block used by MANet_Star."""

    def __init__(self, dim: int, mlp_ratio: int = 3):
        super().__init__()
        inner = int(mlp_ratio * dim)
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, inner, kernel_size=1)
        self.f2 = nn.Conv2d(dim, inner, kernel_size=1)
        self.g = Conv(inner, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        return residual + x


class MANet_Star(MANet):
    """MANet variant that replaces Bottleneck blocks with StarBlock."""

    def __init__(
        self,
        c1: int = 256,
        c2: int = 256,
        n: int = 1,
        shortcut: bool = True,
        p: float = 1.0,
        kernel_size: int = 3,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__(c1=c1, c2=c2, n=n, shortcut=shortcut, p=p, kernel_size=kernel_size, g=g, e=e)
        self.m = nn.ModuleList(StarBlock(self.c) for _ in range(n))


class MANet_FasterBlock(MANet):
    """MANet variant that replaces Bottleneck blocks with Faster_Block."""

    def __init__(
        self,
        c1: int = 256,
        c2: int = 256,
        n: int = 1,
        shortcut: bool = True,
        p: float = 1.0,
        kernel_size: int = 3,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__(c1=c1, c2=c2, n=n, shortcut=shortcut, p=p, kernel_size=kernel_size, g=g, e=e)
        from ultralytics.nn.extraction.c3k2_base import Faster_Block

        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))


class MANet_FasterCGLU(MANet):
    """MANet variant that replaces Bottleneck blocks with Faster_Block_CGLU."""

    def __init__(
        self,
        c1: int = 256,
        c2: int = 256,
        n: int = 1,
        shortcut: bool = True,
        p: float = 1.0,
        kernel_size: int = 3,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__(c1=c1, c2=c2, n=n, shortcut=shortcut, p=p, kernel_size=kernel_size, g=g, e=e)
        from ultralytics.nn.extraction.c3k2_base import Faster_Block_CGLU

        self.m = nn.ModuleList(Faster_Block_CGLU(self.c, self.c) for _ in range(n))


__all__ = ["HyperComputeModule", "MANet", "MANet_Star", "MANet_FasterBlock", "MANet_FasterCGLU"]
