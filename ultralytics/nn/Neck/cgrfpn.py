"""CGRFPN building blocks migrated from RTDETR-main (minimal dependency version).

Modules:
- PyramidContextExtraction
- GetIndexOutput
- RCM
- FuseBlockMulti
- DynamicInterpolationFusion
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

try:
    from timm.layers import DropPath
except Exception:  # pragma: no cover
    class DropPath(nn.Identity):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()


class PyramidPoolAgg_PCE(nn.Module):
    """Adaptive pyramid pooling and concat for multi-level inputs."""

    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        _, _, h, w = inputs[-1].shape
        h = (h - 1) // self.stride + 1
        w = (w - 1) // self.stride + 1
        return torch.cat([F.adaptive_avg_pool2d(inp, (h, w)) for inp in inputs], dim=1)


class h_sigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3) / 6


class ConvMlp(nn.Module):
    """MLP implemented by 1x1 convolutions."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.ReLU,
        norm_layer=None,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class RCA(nn.Module):
    """Rectangular Context Attention unit used in RCM."""

    def __init__(
        self,
        inp: int,
        ratio: int = 2,
        band_kernel_size: int = 11,
        square_kernel_size: int = 3,
    ):
        super().__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        gc = max(inp // ratio, 1)
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loc = self.dwconv_hw(x)
        att = self.excite(self.pool_h(x) + self.pool_w(x))
        return att * loc


class RCM(nn.Module):
    """Rectangular Self-Calibration Module (ECCV'24 style)."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 2.0,
        act_layer: type[nn.Module] = nn.GELU,
        ls_init_value: float = 1e-6,
        drop_path: float = 0.0,
        dw_size: int = 11,
        square_kernel_size: int = 3,
        ratio: int = 1,
    ):
        super().__init__()
        self.token_mixer = RCA(dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size, ratio=ratio)
        self.norm = nn.BatchNorm2d(dim)
        self.mlp = ConvMlp(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return self.drop_path(x) + shortcut


class PyramidContextExtraction(nn.Module):
    """P3/P4/P5 context extraction and split."""

    def __init__(self, dim: list[int] | tuple[int, ...], n: int = 3):
        super().__init__()
        self.dim = list(dim)
        self.ppa = PyramidPoolAgg_PCE()
        self.rcm = nn.Sequential(*[RCA(sum(self.dim), ratio=2, band_kernel_size=3, square_kernel_size=1) for _ in range(n)])

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        x = self.ppa(x)
        x = self.rcm(x)
        return torch.split(x, self.dim, dim=1)


class GetIndexOutput(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


class FuseBlockMulti(nn.Module):
    def __init__(self, inp: int):
        super().__init__()
        self.fuse1 = Conv(inp, inp, act=False)
        self.fuse2 = Conv(inp, inp, act=False)
        self.act = h_sigmoid()

    def forward(self, x):
        x_l, x_h = x
        _, _, h, w = x_l.shape
        inp = self.fuse1(x_l)
        sig_act = self.fuse2(x_h)
        sig_act = F.interpolate(self.act(sig_act), size=(h, w), mode="bilinear", align_corners=False)
        return inp * sig_act


class DynamicInterpolationFusion(nn.Module):
    def __init__(self, chn: list[int] | tuple[int, ...]):
        super().__init__()
        self.conv = nn.Conv2d(chn[1], chn[0], kernel_size=1)

    def forward(self, x):
        return x[0] + self.conv(F.interpolate(x[1], size=x[0].size()[2:], mode="bilinear", align_corners=False))

