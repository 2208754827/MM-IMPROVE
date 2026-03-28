import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class SpatialDependencyPerception(nn.Module):
    """Stable SDFM-compatible fusion block.

    Keeps the same YAML/API usage as SDFM while using a numerically stable
    gated fusion path for training robustness in mixed precision.
    """

    def __init__(self, dim: int | None = None, patch: int = 8, inter_dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.patch = int(patch)
        self.inter_dim = inter_dim
        self._built = False
        self._c = None
        self.reduce: nn.Module | None = None
        self.dw: nn.Module | None = None
        self.gate: nn.Module | None = None
        self.res_scale = 0.2
        if isinstance(dim, int) and dim > 0:
            self._build(dim)

    def _build(self, c: int) -> None:
        ci = c if self.inter_dim is None else int(self.inter_dim)
        self.reduce = nn.Sequential(
            nn.Conv2d(c * 2, ci, 1, bias=False),
            nn.GroupNorm(min(32, ci), ci),
            nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(ci, ci, 3, stride=1, padding=1, groups=ci, bias=False),
            nn.GroupNorm(min(32, ci), ci),
            nn.SiLU(inplace=True),
            nn.Conv2d(ci, c, 1, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=True),
            nn.Sigmoid(),
        )
        self._built = True
        self._c = c

    def _build_if_needed(self, c: int, device: torch.device | None = None) -> None:
        if not (self._built and self._c == c):
            self._build(c)
        if device is not None:
            self.reduce = self.reduce.to(device=device)
            self.dw = self.dw.to(device=device)
            self.gate = self.gate.to(device=device)

    def forward(self, x_low, x_high=None):
        if x_high is None and isinstance(x_low, (list, tuple)):
            x_low, x_high = x_low

        if not isinstance(x_low, torch.Tensor) or not isinstance(x_high, torch.Tensor):
            raise TypeError("SpatialDependencyPerception expects two input tensors")
        if x_low.shape != x_high.shape:
            raise ValueError(f"SDFM expects equal shapes, got {x_low.shape} vs {x_high.shape}")

        b, c, h, w = x_low.shape
        if h % self.patch != 0 or w % self.patch != 0:
            raise ValueError(f"SDFM expects H/W divisible by patch={self.patch}, got {(h, w)}")

        out_dtype = x_low.dtype
        with autocast(enabled=False):
            self._build_if_needed(c, x_low.device)
            weight_dtype = self.reduce[0].weight.dtype
            x_low_f = x_low.to(dtype=weight_dtype)
            x_high_f = x_high.to(dtype=weight_dtype)

            fused = torch.cat([x_low_f, x_high_f], dim=1)
            fused = self.reduce(fused)
            fused = self.dw(fused)
            g = self.gate(x_high_f - x_low_f)
            out = x_low_f + self.res_scale * fused * g
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        return out.to(dtype=out_dtype)
