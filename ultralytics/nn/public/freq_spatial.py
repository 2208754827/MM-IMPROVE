"""FreqSpatial blocks migrated from RTDETR-20260203."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from ultralytics.nn.modules.conv import Conv

__all__ = ("ScharrConv", "FreqSpatial")


class ScharrConv(nn.Module):
    """Depthwise Scharr edge extractor."""

    def __init__(self, channels: int):
        super().__init__()
        kx = torch.tensor(
            [[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[3.0, 10.0, 3.0], [0.0, 0.0, 0.0], [-3.0, -10.0, -3.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.conv_x = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv_y = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv_x.weight.data.copy_(kx.expand(channels, 1, 3, 3))
        self.conv_y.weight.data.copy_(ky.expand(channels, 1, 3, 3))
        self.conv_x.weight.requires_grad_(False)
        self.conv_y.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        return 0.5 * gx + 0.5 * gy


class FreqSpatial(nn.Module):
    """Spatial edge branch + frequency branch fusion."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.sed = ScharrConv(in_channels)
        self.spatial_conv1 = Conv(in_channels, in_channels)
        self.spatial_conv2 = Conv(in_channels, in_channels)

        self.fft_conv = Conv(in_channels * 2, in_channels * 2, 3)
        self.fft_conv2 = Conv(in_channels, in_channels, 3)
        self.final_conv = Conv(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.shape
        x_dtype = x.dtype

        spatial_feat = self.sed(x)
        spatial_feat = self.spatial_conv1(spatial_feat)
        spatial_feat = self.spatial_conv2(spatial_feat + x)

        # FFT ops run in FP32 to avoid ComplexHalf deterministic runtime errors under AMP.
        # Convs run in module dtype (fp16/fp32) to keep validation/inference dtype-compatible.
        with autocast(enabled=False):
            x_fft = x.float()
            fft_feat = torch.fft.rfft2(x_fft, norm="ortho")
            fft_real = torch.real(fft_feat).unsqueeze(-1)
            fft_imag = torch.imag(fft_feat).unsqueeze(-1)
            fft_feat = torch.cat((fft_real, fft_imag), dim=-1)  # [B, C, H, Wf, 2]
            fft_feat = fft_feat.permute(0, 1, 4, 2, 3).contiguous().view(x.shape[0], c * 2, h, -1)

        conv1_dtype = self.fft_conv.conv.weight.dtype
        fft_feat = self.fft_conv(fft_feat.to(dtype=conv1_dtype))

        with autocast(enabled=False):
            wf = fft_feat.shape[-1]
            fft_feat = fft_feat.float().view(x.shape[0], c, 2, h, wf).permute(0, 1, 3, 4, 2).contiguous()
            fft_feat = torch.view_as_complex(fft_feat)
            fft_feat = torch.fft.irfft2(fft_feat, s=(h, w), norm="ortho")

        conv2_dtype = self.fft_conv2.conv.weight.dtype
        fft_feat = self.fft_conv2(fft_feat.to(dtype=conv2_dtype))
        fft_feat = fft_feat.to(dtype=x_dtype)

        out = spatial_feat + fft_feat
        return self.final_conv(out)
