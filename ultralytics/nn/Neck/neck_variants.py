"""
Neck Modules (AFPN / HS-FPN / CFPT / Fusion / RepPAN / GFPN / Multi-Branch FPN)
迁移自 upstream `ultralytics/nn/extra_modules`，仅包含纯 PyTorch 实现。
"""

from __future__ import annotations

import math
import warnings
from typing import List, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
try:
    import torch_dct as DCT
except Exception:  # 可选依赖缺失时自动降级
    DCT = None
from timm.layers import trunc_normal_, to_2tuple

from ultralytics.nn.modules.conv import Conv, DSConv, autopad
from ultralytics.nn.modules.block import C2f, C3k2, RepConv

__all__ = [
    # AFPN / ASFF
    "GSConv",
    "VoVGSCSP",
    "AFPN_P345",
    "AFPN_P345_Custom",
    "AFPN_P2345",
    "AFPN_P2345_Custom",
    "Zoom_cat",
    "ScalSeq",
    "DynamicScalSeq",
    # HS-FPN
    "HFP",
    "SDP",
    "SDP_Improved",
    "ChannelAttention_HSFPN",
    "ELA_HSFPN",
    "CA_HSFPN",
    "CAA_HSFPN",
    "FSA",
    # CFPT
    "CrossLayerSpatialAttention",
    "CrossLayerChannelAttention",
    "CrossAttentionBlock",
    "Add",
    "Multiply",
    # 频域融合
    "FreqFusion",
    "LocalSimGuidedSampler",
    "PSFM",
    # BIFPN / 加权融合
    "Fusion",
    "GDSAFusion",
    "GLSA",
    "CAFMFusion",
    "SDI",
    # GFPN / RepPAN
    "CSPStage",
    "BiFusion",
    "OREPANCSPELAN4",
    # Re-Calibration FPN
    "SBA",
    # DPCF
    "DPCF",
    # HyperACE
    "HyperACE",
    "FullPAD_Tunnel",
    # Efficient Multi-Branch & Scale FPN
    "EUCB",
    "EUCB_SC",
    "MSDC",
    "MSCB",
    "MSCB_SC",
    "CSP_MSCB",
    "CSP_MSCB_SC",
]

# ---------------- 基础块 ----------------


class GSConv(nn.Module):
    """GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = x2.size()
        y = x2.reshape(b * n // 2, 2, h * w).permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False),
        )
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class VoVGSCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


class SDI(nn.Module):
    """Semantics and Detail Infusion"""

    def __init__(self, channels):
        super().__init__()
        self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[2:]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, target_size)
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
            ans = ans * self.convs[i](x)
        return ans


class Fusion(nn.Module):
    """可选 weight/adaptive/concat/bifpn/SDI"""

    def __init__(self, inc_list, fusion="bifpn") -> None:
        super().__init__()
        assert fusion in ["weight", "adaptive", "concat", "bifpn", "SDI"]
        self.fusion = fusion

        if self.fusion == "bifpn":
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == "SDI":
            self.SDI = SDI(inc_list)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])
            if self.fusion == "adaptive":
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)

    def forward(self, x):
        if self.fusion in ["weight", "adaptive"]:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == "weight":
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == "adaptive":
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == "concat":
            return torch.cat(x, dim=1)
        elif self.fusion == "bifpn":
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == "SDI":
            return self.SDI(x)


class ContextBlock(nn.Module):
    """Lightweight context modeling block used by GLSA."""

    def __init__(self, inplanes, ratio=2.0, pooling_type="att", fusion_types=("channel_mul",)):
        super().__init__()
        assert pooling_type in ["avg", "att"]
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ["channel_add", "channel_mul"]
        assert all(f in valid_fusion_types for f in fusion_types)
        assert len(fusion_types) > 0

        self.inplanes = inplanes
        self.planes = max(int(inplanes * ratio), 1)
        self.pooling_type = pooling_type

        if pooling_type == "att":
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_add_conv = None
        self.channel_mul_conv = None
        if "channel_add" in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, inplanes, kernel_size=1),
            )
        if "channel_mul" in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, inplanes, kernel_size=1),
            )
        self._reset_parameters()

    def _reset_parameters(self):
        if self.pooling_type == "att":
            nn.init.kaiming_normal_(self.conv_mask.weight, mode="fan_in")
            if self.conv_mask.bias is not None:
                nn.init.zeros_(self.conv_mask.bias)
        for conv_seq in (self.channel_add_conv, self.channel_mul_conv):
            if conv_seq is not None:
                nn.init.zeros_(conv_seq[-1].weight)
                if conv_seq[-1].bias is not None:
                    nn.init.zeros_(conv_seq[-1].bias)

    def _spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == "att":
            input_x = x.view(batch, channel, height * width).unsqueeze(1)
            context_mask = self.conv_mask(x).view(batch, 1, height * width)
            context_mask = self.softmax(context_mask).unsqueeze(-1)
            context = torch.matmul(input_x, context_mask).view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self._spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            out = out + out * torch.sigmoid(self.channel_mul_conv(context))
        if self.channel_add_conv is not None:
            out = out + self.channel_add_conv(context)
        return out


class GLSAConvBranch(nn.Module):
    """Local branch of GLSA with depthwise spatial mixing."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv(channels, channels, 1, act=nn.ReLU(inplace=True))
        self.dw1 = Conv(channels, channels, 3, g=channels, act=nn.ReLU(inplace=True))
        self.pw1 = Conv(channels, channels, 1, act=nn.ReLU(inplace=True))
        self.dw2 = Conv(channels, channels, 3, g=channels, act=nn.ReLU(inplace=True))
        self.pw2 = Conv(channels, channels, 1, act=nn.SiLU(inplace=True))
        self.gate = nn.Sequential(nn.Conv2d(channels, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.conv1(x)
        y = y + self.dw1(y)
        y = self.pw1(y)
        y = y + self.dw2(y)
        y = self.pw2(y)
        return x * (1.0 + self.gate(y))


class GLSA(nn.Module):
    """Global-to-Local Spatial Aggregation block."""

    def __init__(self, c1, c2, context_ratio=2.0):
        super().__init__()
        self.pre = Conv(c1, c2, 1)
        if c2 <= 1:
            self.local_channels = c2
            self.global_channels = 0
            self.local_branch = GLSAConvBranch(c2)
            self.global_block = None
        else:
            self.local_channels = max(1, c2 // 2)
            self.global_channels = c2 - self.local_channels
            self.local_branch = GLSAConvBranch(self.local_channels)
            self.global_block = ContextBlock(
                self.global_channels, ratio=max(float(context_ratio), 1.0), pooling_type="att", fusion_types=("channel_mul",)
            )
        self.fuse = Conv(c2, c2, 1)

    def forward(self, x):
        x = self.pre(x)
        if self.global_channels == 0:
            return self.fuse(self.local_branch(x))

        local_x = x[:, : self.local_channels, :, :]
        global_x = x[:, self.local_channels :, :, :]
        local_feat = self.local_branch(local_x)
        global_feat = self.global_block(global_x)
        return self.fuse(torch.cat((local_feat, global_feat), dim=1))


class GEFM(nn.Module):
    """Guided enhancement fusion used by PSFM."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.rgb_k = DSConv(out_c, out_c, 3)
        self.rgb_v = DSConv(out_c, out_c, 3)
        self.q = DSConv(in_c, out_c, 3)
        self.inf_k = DSConv(out_c, out_c, 3)
        self.inf_v = DSConv(out_c, out_c, 3)
        self.second_reduce = DSConv(in_c, out_c, 3)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        q = self.q(torch.cat([x, y], dim=1))
        rgb_k = self.rgb_k(x)
        rgb_v = self.rgb_v(x)
        b, _, h, w = rgb_v.size()
        rgb_v = rgb_v.view(b, -1, w * h)
        rgb_k = rgb_k.view(b, -1, w * h).permute(0, 2, 1)
        rgb_q = q.view(b, -1, w * h)
        rgb_mask = self.softmax(torch.bmm(rgb_k, rgb_q))
        rgb_refine = torch.bmm(rgb_v, rgb_mask.permute(0, 2, 1)).view(b, -1, h, w)
        rgb_refine = self.gamma1 * rgb_refine + y

        inf_k = self.inf_k(y)
        inf_v = self.inf_v(y)
        inf_v = inf_v.view(b, -1, w * h)
        inf_k = inf_k.view(b, -1, w * h).permute(0, 2, 1)
        inf_q = q.view(b, -1, w * h)
        inf_mask = self.softmax(torch.bmm(inf_k, inf_q))
        inf_refine = torch.bmm(inf_v, inf_mask.permute(0, 2, 1)).view(b, -1, h, w)
        inf_refine = self.gamma2 * inf_refine + x
        return self.second_reduce(torch.cat([rgb_refine, inf_refine], dim=1))


class DenseLayer(nn.Module):
    """Dense refinement block used by PSFM."""

    def __init__(self, in_c, out_c, down_factor=4, k=2):
        super().__init__()
        mid_c = out_c // down_factor
        self.down = nn.Conv2d(in_c, mid_c, 1)
        self.denseblock = nn.ModuleList([DSConv(mid_c * i, mid_c, 3) for i in range(1, k + 1)])
        self.fuse = DSConv(in_c + mid_c, out_c, 3)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for block in self.denseblock:
            feats = block(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class PSFM(nn.Module):
    """Profound Semantic Fusion Module."""

    def __init__(self, channel):
        super().__init__()
        self.rgb_obj = DenseLayer(channel, channel)
        self.inf_obj = DenseLayer(channel, channel)
        self.obj_fuse = GEFM(channel * 2, channel)

    def forward(self, data):
        rgb, depth = data
        rgb_sum = self.rgb_obj(rgb)
        inf_sum = self.inf_obj(depth)
        return self.obj_fuse(rgb_sum, inf_sum)


class PixelAttention_CGA(nn.Module):
    """Pixel attention used by CAFM-based fusion."""

    def __init__(self, dim):
        super().__init__()
        self.pa = nn.Conv2d(2 * dim, dim, 7, padding=3, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn):
        x = x.unsqueeze(2)  # (B,C,1,H,W)
        pattn = pattn.unsqueeze(2)  # (B,C,1,H,W)
        x2 = torch.cat([x, pattn], dim=2)  # (B,C,2,H,W)
        x2 = einops.rearrange(x2, "b c t h w -> b (c t) h w")
        return self.sigmoid(self.pa(x2))


class CAFM(nn.Module):
    """Convolution and attention fusion module."""

    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        assert dim % num_heads == 0, f"CAFM requires dim({dim}) divisible by num_heads({num_heads})"
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // num_heads, dim, kernel_size=(3, 3, 3), padding=1, groups=dim // num_heads, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x3 = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x3)).squeeze(2)  # (B,3C,H,W)

        # Local conv branch.
        f_all = qkv.reshape(b, h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2)).squeeze(2)
        f_conv = f_all.permute(0, 3, 1, 2).reshape(b, 9 * c // self.num_heads, h, w).unsqueeze(2)
        out_conv = self.dep_conv(f_conv).squeeze(2)

        # Global self-attention branch.
        q, k, v = qkv.chunk(3, dim=1)
        q = einops.rearrange(q, "b (head cc) hh ww -> b head cc (hh ww)", head=self.num_heads)
        k = einops.rearrange(k, "b (head cc) hh ww -> b head cc (hh ww)", head=self.num_heads)
        v = einops.rearrange(v, "b (head cc) hh ww -> b head cc (hh ww)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = einops.rearrange(out, "b head cc (hh ww) -> b (head cc) hh ww", head=self.num_heads, hh=h, ww=w)
        out = self.project_out(out.unsqueeze(2)).squeeze(2)
        return out + out_conv


class CAFMFusion(nn.Module):
    """CAFM-guided two-branch feature fusion."""

    def __init__(self, dim, heads=8):
        super().__init__()
        self.cafm = CAFM(dim, num_heads=heads)
        self.pa = PixelAttention_CGA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data
        initial = x + y
        pattn1 = self.cafm(initial)
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        out = initial + pattn2 * x + (1.0 - pattn2) * y
        return self.conv(out)


# ---------------- AFPN / ASFF ----------------


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, filter_in, filter_out):
        super().__init__()
        self.conv1 = Conv(filter_in, filter_out, 3)
        self.conv2 = Conv(filter_out, filter_out, 3, act=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return self.conv1.act(out)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Sequential(Conv(in_channels, out_channels, 1), nn.Upsample(scale_factor=scale_factor, mode="bilinear"))

    def forward(self, x):
        return self.upsample(x)


class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = Conv(in_channels, out_channels, 2, 2, 0)

    def forward(self, x):
        return self.downsample(x)


class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = Conv(in_channels, out_channels, 4, 4, 0)

    def forward(self, x):
        return self.downsample(x)


class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = Conv(in_channels, out_channels, 8, 8, 0)

    def forward(self, x):
        return self.downsample(x)


class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super().__init__()
        compress_c = 8
        self.weight_level_1 = Conv(inter_dim, compress_c, 1)
        self.weight_level_2 = Conv(inter_dim, compress_c, 1)
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(inter_dim, inter_dim, 3)

    def forward(self, input1, input2):
        w1 = self.weight_level_1(input1)
        w2 = self.weight_level_2(input2)
        levels_weight = self.weight_levels(torch.cat((w1, w2), 1))
        levels_weight = F.softmax(levels_weight, dim=1)
        fused = input1 * levels_weight[:, 0:1] + input2 * levels_weight[:, 1:2]
        return self.conv(fused)


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super().__init__()
        compress_c = 8
        self.weight_level_1 = Conv(inter_dim, compress_c, 1)
        self.weight_level_2 = Conv(inter_dim, compress_c, 1)
        self.weight_level_3 = Conv(inter_dim, compress_c, 1)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(inter_dim, inter_dim, 3)

    def forward(self, input1, input2, input3):
        w1 = self.weight_level_1(input1)
        w2 = self.weight_level_2(input2)
        w3 = self.weight_level_3(input3)
        levels_weight = self.weight_levels(torch.cat((w1, w2, w3), 1))
        levels_weight = F.softmax(levels_weight, dim=1)
        fused = input1 * levels_weight[:, 0:1] + input2 * levels_weight[:, 1:2] + input3 * levels_weight[:, 2:]
        return self.conv(fused)


class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512):
        super().__init__()
        compress_c = 8
        self.weight_level_0 = Conv(inter_dim, compress_c, 1)
        self.weight_level_1 = Conv(inter_dim, compress_c, 1)
        self.weight_level_2 = Conv(inter_dim, compress_c, 1)
        self.weight_level_3 = Conv(inter_dim, compress_c, 1)
        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(inter_dim, inter_dim, 3)

    def forward(self, input0, input1, input2, input3):
        w0 = self.weight_level_0(input0)
        w1 = self.weight_level_1(input1)
        w2 = self.weight_level_2(input2)
        w3 = self.weight_level_3(input3)
        levels_weight = self.weight_levels(torch.cat((w0, w1, w2, w3), 1))
        levels_weight = F.softmax(levels_weight, dim=1)
        fused = input0 * levels_weight[:, 0:1] + input1 * levels_weight[:, 1:2] + input2 * levels_weight[:, 2:3] + input3 * levels_weight[:, 3:]
        return self.conv(fused)


class BlockBody_P345(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        channels = list(channels)

        self.blocks_scalezero1 = nn.Sequential(Conv(channels[0], channels[0], 1))
        self.blocks_scaleone1 = nn.Sequential(Conv(channels[1], channels[1], 1))
        self.blocks_scaletwo1 = nn.Sequential(Conv(channels[2], channels[2], 1))

        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        self.blocks_scalezero2 = nn.Sequential(*[BasicBlock(channels[0], channels[0]) for _ in range(4)])
        self.blocks_scaleone2 = nn.Sequential(*[BasicBlock(channels[1], channels[1]) for _ in range(4)])

        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])

        self.blocks_scalezero3 = nn.Sequential(*[BasicBlock(channels[0], channels[0]) for _ in range(4)])
        self.blocks_scaleone3 = nn.Sequential(*[BasicBlock(channels[1], channels[1]) for _ in range(4)])
        self.blocks_scaletwo3 = nn.Sequential(*[BasicBlock(channels[2], channels[2]) for _ in range(4)])

        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)

    def forward(self, x):
        x0, x1, x2 = x

        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)

        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)

        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)

        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)

        return x0, x1, x2


class BlockBody_P345_Custom(BlockBody_P345):
    def __init__(self, channels=(64, 128, 256, 512), block_type: object = "C2f"):
        super().__init__(channels)
        block = block_type
        if isinstance(block_type, str):
            if block_type not in globals():
                raise ValueError(f"AFPN Custom block_type 未注册：{block_type}")
            block = globals()[block_type]

        channels = list(channels)
        self.blocks_scalezero2 = block(channels[0], channels[0])
        self.blocks_scaleone2 = block(channels[1], channels[1])
        self.blocks_scalezero3 = block(channels[0], channels[0])
        self.blocks_scaleone3 = block(channels[1], channels[1])
        self.blocks_scaletwo3 = block(channels[2], channels[2])


class BlockBody_P2345(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        channels = list(channels)

        self.blocks_scalezero1 = nn.Sequential(Conv(channels[0], channels[0], 1))
        self.blocks_scaleone1 = nn.Sequential(Conv(channels[1], channels[1], 1))
        self.blocks_scaletwo1 = nn.Sequential(Conv(channels[2], channels[2], 1))
        self.blocks_scalethree1 = nn.Sequential(Conv(channels[3], channels[3], 1))

        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        self.blocks_scalezero2 = nn.Sequential(*[BasicBlock(channels[0], channels[0]) for _ in range(4)])
        self.blocks_scaleone2 = nn.Sequential(*[BasicBlock(channels[1], channels[1]) for _ in range(4)])

        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])

        self.blocks_scalezero3 = nn.Sequential(*[BasicBlock(channels[0], channels[0]) for _ in range(4)])
        self.blocks_scaleone3 = nn.Sequential(*[BasicBlock(channels[1], channels[1]) for _ in range(4)])
        self.blocks_scaletwo3 = nn.Sequential(*[BasicBlock(channels[2], channels[2]) for _ in range(4)])

        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scalezero3_8 = Downsample_x8(channels[0], channels[3])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaleone3_4 = Downsample_x4(channels[1], channels[3])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo3_2 = Downsample_x2(channels[2], channels[3])
        self.upsample_scalethree3_8 = Upsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree3_4 = Upsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree3_2 = Upsample(channels[3], channels[2], scale_factor=2)

        self.asff_scalezero3 = ASFF_4(inter_dim=channels[0])
        self.asff_scaleone3 = ASFF_4(inter_dim=channels[1])
        self.asff_scaletwo3 = ASFF_4(inter_dim=channels[2])
        self.asff_scalethree3 = ASFF_4(inter_dim=channels[3])

        self.blocks_scalezero4 = nn.Sequential(*[BasicBlock(channels[0], channels[0]) for _ in range(4)])
        self.blocks_scaleone4 = nn.Sequential(*[BasicBlock(channels[1], channels[1]) for _ in range(4)])
        self.blocks_scaletwo4 = nn.Sequential(*[BasicBlock(channels[2], channels[2]) for _ in range(4)])
        self.blocks_scalethree4 = nn.Sequential(*[BasicBlock(channels[3], channels[3]) for _ in range(4)])

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)
        x3 = self.blocks_scalethree1(x3)

        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)

        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)

        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)

        scalezero = self.asff_scalezero3(
            x0,
            self.upsample_scaleone3_2(x1),
            self.upsample_scaletwo3_4(x2),
            self.upsample_scalethree3_8(x3),
        )
        scaleone = self.asff_scaleone3(
            self.downsample_scalezero3_2(x0),
            x1,
            self.upsample_scaletwo3_2(x2),
            self.upsample_scalethree3_4(x3),
        )
        scaletwo = self.asff_scaletwo3(
            self.downsample_scalezero3_4(x0),
            self.downsample_scaleone3_2(x1),
            x2,
            self.upsample_scalethree3_2(x3),
        )
        scalethree = self.asff_scalethree3(
            self.downsample_scalezero3_8(x0),
            self.downsample_scaleone3_4(x1),
            self.downsample_scaletwo3_2(x2),
            x3,
        )

        scalezero = self.blocks_scalezero4(scalezero)
        scaleone = self.blocks_scaleone4(scaleone)
        scaletwo = self.blocks_scaletwo4(scaletwo)
        scalethree = self.blocks_scalethree4(scalethree)

        return scalezero, scaleone, scaletwo, scalethree


class BlockBody_P2345_Custom(BlockBody_P2345):
    def __init__(self, channels=(64, 128, 256, 512), block_type: object = "C2f"):
        super().__init__(channels)
        block = block_type
        if isinstance(block_type, str):
            if block_type not in globals():
                raise ValueError(f"AFPN Custom block_type 未注册：{block_type}")
            block = globals()[block_type]

        channels = list(channels)
        self.blocks_scalezero2 = block(channels[0], channels[0])
        self.blocks_scaleone2 = block(channels[1], channels[1])

        self.blocks_scalezero3 = block(channels[0], channels[0])
        self.blocks_scaleone3 = block(channels[1], channels[1])
        self.blocks_scaletwo3 = block(channels[2], channels[2])

        self.blocks_scalezero4 = block(channels[0], channels[0])
        self.blocks_scaleone4 = block(channels[1], channels[1])
        self.blocks_scaletwo4 = block(channels[2], channels[2])
        self.blocks_scalethree4 = block(channels[3], channels[3])


class AFPN_P345(nn.Module):
    def __init__(self, ch=(256, 512, 1024), hidc: int = 256, factor: int = 4):
        super().__init__()
        ch = list(ch)

        self.conv0 = Conv(ch[0], ch[0] // factor, 1)
        self.conv1 = Conv(ch[1], ch[1] // factor, 1)
        self.conv2 = Conv(ch[2], ch[2] // factor, 1)

        self.body = nn.Sequential(BlockBody_P345([ch[0] // factor, ch[1] // factor, ch[2] // factor]))

        self.conv00 = Conv(ch[0] // factor, hidc, 1)
        self.conv11 = Conv(ch[1] // factor, hidc, 1)
        self.conv22 = Conv(ch[2] // factor, hidc, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        out0, out1, out2 = self.body([x0, x1, x2])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        return [out0, out1, out2]


class AFPN_P345_Custom(nn.Module):
    def __init__(self, ch=(256, 512, 1024), hidc: int = 256, block_type: object = "C2f", factor: int = 4):
        super().__init__()
        ch = list(ch)

        self.conv0 = Conv(ch[0], ch[0] // factor, 1)
        self.conv1 = Conv(ch[1], ch[1] // factor, 1)
        self.conv2 = Conv(ch[2], ch[2] // factor, 1)

        self.body = nn.Sequential(
            BlockBody_P345_Custom([ch[0] // factor, ch[1] // factor, ch[2] // factor], block_type=block_type)
        )

        self.conv00 = Conv(ch[0] // factor, hidc, 1)
        self.conv11 = Conv(ch[1] // factor, hidc, 1)
        self.conv22 = Conv(ch[2] // factor, hidc, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        out0, out1, out2 = self.body([x0, x1, x2])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        return [out0, out1, out2]


class AFPN_P2345(nn.Module):
    def __init__(self, ch=(256, 512, 1024, 2048), hidc: int = 256, factor: int = 4):
        super().__init__()
        ch = list(ch)

        self.conv0 = Conv(ch[0], ch[0] // factor, 1)
        self.conv1 = Conv(ch[1], ch[1] // factor, 1)
        self.conv2 = Conv(ch[2], ch[2] // factor, 1)
        self.conv3 = Conv(ch[3], ch[3] // factor, 1)

        self.body = nn.Sequential(
            BlockBody_P2345([ch[0] // factor, ch[1] // factor, ch[2] // factor, ch[3] // factor])
        )

        self.conv00 = Conv(ch[0] // factor, hidc, 1)
        self.conv11 = Conv(ch[1] // factor, hidc, 1)
        self.conv22 = Conv(ch[2] // factor, hidc, 1)
        self.conv33 = Conv(ch[3] // factor, hidc, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)
        return [out0, out1, out2, out3]


class AFPN_P2345_Custom(nn.Module):
    def __init__(self, ch=(256, 512, 1024, 2048), hidc: int = 256, block_type: object = "C2f", factor: int = 4):
        super().__init__()
        ch = list(ch)

        self.conv0 = Conv(ch[0], ch[0] // factor, 1)
        self.conv1 = Conv(ch[1], ch[1] // factor, 1)
        self.conv2 = Conv(ch[2], ch[2] // factor, 1)
        self.conv3 = Conv(ch[3], ch[3] // factor, 1)

        self.body = nn.Sequential(
            BlockBody_P2345_Custom(
                [ch[0] // factor, ch[1] // factor, ch[2] // factor, ch[3] // factor],
                block_type=block_type,
            )
        )

        self.conv00 = Conv(ch[0] // factor, hidc, 1)
        self.conv11 = Conv(ch[1] // factor, hidc, 1)
        self.conv22 = Conv(ch[2] // factor, hidc, 1)
        self.conv33 = Conv(ch[3] // factor, hidc, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)
        return [out0, out1, out2, out3]


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ["lp", "pl"]
        if style == "pl":
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == "pl":
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        if self.offset.bias is not None:
            nn.init.constant_(self.offset.bias, 0.0)

        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            nn.init.constant_(self.scope.weight, 0.0)
            if self.scope.bias is not None:
                nn.init.constant_(self.scope.bias, 0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid(h, h, indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def sample(self, x, offset):
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        coords = (
            torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(x.dtype)
        )
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
            .view(b, 2, -1, self.scale * h, self.scale * w)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(b, -1, self.scale * h, self.scale * w)

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == "pl":
            return self.forward_pl(x)
        return self.forward_lp(x)


class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode="nearest")
        return torch.cat([l, m, s], dim=1)


class ScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super().__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        if hasattr(self, "conv0"):
            p3 = self.conv0(p3)
        p4_2 = F.interpolate(self.conv1(p4), p3.size()[2:], mode="nearest")
        p5_2 = F.interpolate(self.conv2(p5), p3.size()[2:], mode="nearest")
        combine = torch.cat([p3.unsqueeze(-3), p4_2.unsqueeze(-3), p5_2.unsqueeze(-3)], dim=2)
        y = self.pool_3d(self.act(self.bn(self.conv3d(combine))))
        return y.squeeze(2)


class DynamicScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super().__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))
        self.dysample1 = DySample(channel, 2, "lp")
        self.dysample2 = DySample(channel, 4, "lp")

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        if hasattr(self, "conv0"):
            p3 = self.conv0(p3)
        p4_2 = self.dysample1(self.conv1(p4))
        p5_2 = self.dysample2(self.conv2(p5))
        combine = torch.cat([p3.unsqueeze(-3), p4_2.unsqueeze(-3), p5_2.unsqueeze(-3)], dim=2)
        y = self.pool_3d(self.act(self.bn(self.conv3d(combine))))
        return y.squeeze(2)


# ---------------- HS-FPN & 注意力增强 ----------------


class DctSpatialInteraction(nn.Module):
    def __init__(self, in_channels, ratio, isdct=True):
        super().__init__()
        self.ratio = ratio
        self.isdct = isdct
        if (not self.isdct) or DCT is None:
            self.spatial1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if (not self.isdct) or DCT is None:
            return x * torch.sigmoid(self.spatial1x1(x))
        idct = DCT.dct_2d(x, norm="ortho")
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)
        dct = idct * weight
        dct_ = DCT.idct_2d(dct, norm="ortho")
        return x * dct_

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


class DctChannelInteraction(nn.Module):
    def __init__(self, in_channels, patch, ratio, isdct=True):
        super().__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Conv2d(in_channels, in_channels, 1, groups=32)
        self.channel2x1 = nn.Conv2d(in_channels, in_channels, 1, groups=32)
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if (not self.isdct) or DCT is None:
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm="ortho")
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight
        dct_ = DCT.idct_2d(dct, norm="ortho")

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


class HFP(nn.Module):
    def __init__(self, in_channels, ratio=(0.25, 0.25), patch=(8, 8), isdct=True):
        super().__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels),
        )

    def forward(self, x):
        spatial = self.spatial(x)
        channel = self.channel(x)
        return self.out(spatial + channel)


class SDP(nn.Module):
    def __init__(self, in_dim, dim=256, patch_size=None, inter_dim=None):
        super().__init__()
        self.conv1x1_0 = Conv(in_dim[0], dim) if in_dim[0] != dim else nn.Identity()
        self.conv1x1_1 = Conv(in_dim[1], dim) if in_dim[1] != dim else nn.Identity()
        self.inter_dim = inter_dim or dim
        self.conv_q = nn.Sequential(nn.Conv2d(dim, self.inter_dim, 1, bias=False), nn.GroupNorm(32, self.inter_dim))
        self.conv_k = nn.Sequential(nn.Conv2d(dim, self.inter_dim, 1, bias=False), nn.GroupNorm(32, self.inter_dim))
        self.softmax = nn.Softmax(dim=-1)
        self.patch_size = patch_size

    def forward(self, x):
        x_low, x_high = x
        x_low = self.conv1x1_0(x_low)
        x_high = self.conv1x1_1(x_high)
        b_, _, h_, w_ = x_low.size()
        q = einops.rearrange(
            self.conv_q(x_low), "b c (h p1) (w p2) -> (b h w) c (p1 p2)", p1=self.patch_size[0], p2=self.patch_size[1]
        ).transpose(1, 2)
        k = einops.rearrange(
            self.conv_k(x_high), "b c (h p1) (w p2) -> (b h w) c (p1 p2)", p1=self.patch_size[0], p2=self.patch_size[1]
        )
        attn = torch.matmul(q, k) / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1, 2)
        output = torch.matmul(attn, v)
        output = einops.rearrange(
            output.transpose(1, 2).contiguous(),
            "(b h w) c (p1 p2) -> b c (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=h_ // self.patch_size[0],
            w=w_ // self.patch_size[1],
        )
        return output + x_low


class SDP_Improved(nn.Module):
    def __init__(self, dim=256, inter_dim=None):
        super().__init__()
        self.inter_dim = inter_dim or dim
        self.conv_q = nn.Sequential(nn.Conv2d(dim, self.inter_dim, 3, padding=1, bias=False), nn.GroupNorm(32, self.inter_dim))
        self.conv_k = nn.Sequential(nn.Conv2d(dim, self.inter_dim, 3, padding=1, bias=False), nn.GroupNorm(32, self.inter_dim))
        self.conv = nn.Sequential(nn.Conv2d(self.inter_dim, dim, 3, padding=1, bias=False), nn.GroupNorm(32, dim))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_low, x_high, patch_size):
        b_, _, h_, w_ = x_low.size()
        q = einops.rearrange(
            self.conv_q(x_low), "b c (h p1) (w p2) -> (b h w) c (p1 p2)", p1=patch_size[0], p2=patch_size[1]
        ).transpose(1, 2)
        k = einops.rearrange(
            self.conv_k(x_high), "b c (h p1) (w p2) -> (b h w) c (p1 p2)", p1=patch_size[0], p2=patch_size[1]
        )
        attn = torch.matmul(q, k) / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1, 2)
        output = torch.matmul(attn, v)
        output = einops.rearrange(
            output.transpose(1, 2).contiguous(),
            "(b h w) c (p1 p2) -> b c (h p1) (w p2)",
            p1=patch_size[0],
            p2=patch_size[1],
            h=h_ // patch_size[0],
            w=w_ // patch_size[1],
        )
        return self.conv(output) + x_low


class ChannelAttention_HSFPN(nn.Module):
    def __init__(self, c1, ratio=4, flag=True):
        super().__init__()
        if isinstance(ratio, bool):
            flag = ratio
            ratio = 4
        ratio = max(int(ratio), 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(c1 // ratio, 1)
        self.fc = nn.Sequential(Conv(c1, mid, 1), Conv(mid, c1, 1, act=False))
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * out if self.flag else out


class ELA_HSFPN(nn.Module):
    def __init__(self, c1, flag=True):
        super().__init__()
        gn_groups = math.gcd(int(c1), 16)
        gn_groups = max(1, gn_groups)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(c1, c1, 7, padding=3, bias=True),
            nn.GroupNorm(gn_groups, c1),
            nn.Sigmoid(),
        )
        self.flag = flag

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv1x1(self.pool_h(x).reshape(b, c, h)).reshape(b, c, h, 1)
        x_w = self.conv1x1(self.pool_w(x).reshape(b, c, w)).reshape(b, c, 1, w)
        out = x_h * x_w
        return x * out if self.flag else out


class CA_HSFPN(nn.Module):
    def __init__(self, c1, reduction=8, flag=True):
        super().__init__()
        if isinstance(reduction, bool):
            flag = reduction
            reduction = 8
        reduction = max(int(reduction), 1)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.flag = flag

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = a_w * a_h
        return x * out if self.flag else out


class CAA_HSFPN(nn.Module):
    def __init__(self, c1, c2=None, flag=True, h_kernel_size=11, v_kernel_size=11):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(c2, c2, 1)
        self.h_conv = nn.Conv2d(c2, c2, (1, h_kernel_size), 1, (0, h_kernel_size // 2), groups=c2)
        self.v_conv = nn.Conv2d(c2, c2, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), groups=c2)
        self.conv2 = Conv(c2, c2, 1)
        self.act = nn.Sigmoid()
        self.flag = flag

    def forward(self, x):
        x = self.proj(x)
        out = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return out * x if self.flag else out


class AdaptiveGlobalFilter(nn.Module):
    def __init__(self, ratio=10, dim=32, h=512, w=512):
        super().__init__()
        self.ratio = int(ratio)
        self.filter = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32), requires_grad=True)

    def _resize_filter(self, c, h, w, device):
        # Parameter is [C, H, W, 2]. Interpolate spatially and adapt channel count if needed.
        f = self.filter.to(device=device)
        c0, h0, w0, _ = f.shape
        if h0 != h or w0 != w:
            f_ri = f.permute(3, 0, 1, 2).reshape(1, 2 * c0, h0, w0)
            f_ri = F.interpolate(f_ri, size=(h, w), mode="bilinear", align_corners=False)
            f = f_ri.reshape(2, c0, h, w).permute(1, 2, 3, 0).contiguous()
        if c0 != c:
            if c0 > c:
                f = f[:c]
            else:
                rep = math.ceil(c / c0)
                f = f.repeat(rep, 1, 1, 1)[:c]
        return torch.view_as_complex(f.contiguous())

    def forward(self, x):
        b, c, h, w = x.shape
        x_fre = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        weight = self._resize_filter(c, h, w, x.device).unsqueeze(0)

        r = min(max(self.ratio, 1), h // 2, w // 2)
        mask_low = torch.zeros((h, w), device=x.device, dtype=x_fre.dtype)
        mask_low[h // 2 - r : h // 2 + r, w // 2 - r : w // 2 + r] = 1
        mask_high = 1 - mask_low

        x_low = x_fre * mask_low
        x_high = x_fre * mask_high
        x_new = x_low * weight + x_high
        return torch.fft.ifft2(torch.fft.ifftshift(x_new, dim=(-2, -1)), dim=(-2, -1), norm="ortho").real


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return x * out


class FSA(nn.Module):
    """Spatial-frequency attention block from the reference RTDETR FSA variant."""

    def __init__(self, input_channel=64, size=512, ratio=10):
        super().__init__()
        self.agf = AdaptiveGlobalFilter(ratio=ratio, dim=input_channel, h=size, w=size)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.agf(x) + self.sa(x)


class Multiply(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0] * x[1]


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0] + x[1]


# ---------------- Cross-Layer Feature Pyramid Transformer ----------------


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return einops.rearrange(x, "b h w c -> b c h w")


class CrossLayerPosEmbedding3D(nn.Module):
    def __init__(self, num_heads=4, window_size=(5, 3, 1), spatial=True):
        super().__init__()
        self.spatial = spatial
        self.num_heads = num_heads
        self.layer_num = len(window_size)
        if self.spatial:
            self.num_token = sum([i**2 for i in window_size])
            self.num_token_per_level = [i**2 for i in window_size]
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[0] - 1), num_heads))
            coords_h = [torch.arange(ws) - ws // 2 for ws in window_size]
            coords_w = [torch.arange(ws) - ws // 2 for ws in window_size]
            coords_h = [coords_h[i] * window_size[0] / window_size[i] for i in range(len(coords_h) - 1)] + [coords_h[-1]]
            coords_w = [coords_w[i] * window_size[0] / window_size[i] for i in range(len(coords_w) - 1)] + [coords_w[-1]]
            coords = [torch.stack(torch.meshgrid([coord_h, coord_w])) for coord_h, coord_w in zip(coords_h, coords_w)]
            coords_flatten = torch.cat([torch.flatten(coord, 1) for coord in coords], dim=-1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[0] - 1
            relative_coords[:, :, 0] *= 2 * window_size[0] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=0.02)
        else:
            self.num_token = sum([i for i in window_size])
            self.num_token_per_level = [i for i in window_size]
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[0] - 1), num_heads))
            coords_c = [torch.arange(ws) - ws // 2 for ws in window_size]
            coords_c = [coords_c[i] * window_size[0] / window_size[i] for i in range(len(coords_c) - 1)] + [coords_c[-1]]
            coords = torch.cat(coords_c, dim=0)
            coords_flatten = torch.stack([torch.flatten(coord, 0) for coord in coords], dim=-1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.absolute_position_bias = nn.Parameter(torch.zeros(len(window_size), num_heads, 1, 1, 1))
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self):
        pos_indicies = self.relative_position_index.view(-1)
        pos_indicies_floor = torch.floor(pos_indicies).long()
        pos_indicies_ceil = torch.ceil(pos_indicies).long()
        value_floor = self.relative_position_bias_table[pos_indicies_floor]
        value_ceil = self.relative_position_bias_table[pos_indicies_ceil]
        weights_ceil = pos_indicies - pos_indicies_floor.float()
        weights_floor = 1.0 - weights_ceil

        pos_embed = weights_floor.unsqueeze(-1) * value_floor + weights_ceil.unsqueeze(-1) * value_ceil
        pos_embed = pos_embed.reshape(1, 1, self.num_token, -1, self.num_heads).permute(0, 4, 1, 2, 3)
        pos_embed = pos_embed.split(self.num_token_per_level, 3)
        layer_embed = self.absolute_position_bias.split([1 for _ in range(self.layer_num)], 0)
        pos_embed = torch.cat([i + j for (i, j) in zip(pos_embed, layer_embed)], dim=-2)
        return pos_embed


class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3, act=True):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, to_2tuple(k), to_2tuple(1), to_2tuple(k // 2), groups=dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return x + self.activation(self.proj(x))


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def overlaped_window_partition(x, window_size, stride, pad):
    B, C, H, W = x.shape
    out = F.unfold(x, kernel_size=(window_size, window_size), stride=stride, padding=pad)
    return out.reshape(B, C, window_size * window_size, -1).permute(0, 3, 2, 1)


def overlaped_window_reverse(x, H, W, window_size, stride, padding):
    B, Wm, Wsm, C = x.shape
    Ws, S, P = window_size, stride, padding
    x = x.permute(0, 3, 2, 1).reshape(B, C * Wsm, Wm)
    return F.fold(x, output_size=(H, W), kernel_size=(Ws, Ws), padding=P, stride=S)


def overlaped_channel_partition(x, window_size, stride, pad):
    B, HW, C, _ = x.shape
    out = F.unfold(x, kernel_size=(window_size, 1), stride=(stride, 1), padding=(pad, 0))
    return out.reshape(B, HW, window_size, -1)


def overlaped_channel_reverse(x, window_size, stride, pad, outC):
    B, C, Ws, HW = x.shape
    x = x.permute(0, 3, 2, 1).reshape(B, HW * Ws, C)
    return F.fold(x, output_size=(outC, 1), kernel_size=(window_size, 1), padding=(pad, 0), stride=(stride, 1))


class CrossLayerSpatialAttention(nn.Module):
    def __init__(self, in_dim, layer_num=3, beta=1, num_heads=4, mlp_ratio=2, reduction=4):
        super().__init__()
        assert beta % 2 != 0, "beta must be odd"
        self.num_heads = num_heads
        self.reduction = reduction
        self.window_sizes = [(2**i + beta) if i != 0 else (2**i + beta - 1) for i in range(layer_num)][::-1]
        self.token_num_per_layer = [i**2 for i in self.window_sizes]
        self.token_num = sum(self.token_num_per_layer)

        self.stride_list = [2**i for i in range(layer_num)][::-1]
        self.padding_list = [[0, 0] for _ in self.window_sizes]
        self.shape_list = [[0, 0] for _ in range(layer_num)]

        self.hidden_dim = in_dim // reduction
        self.head_dim = self.hidden_dim // num_heads

        self.cpe = nn.ModuleList(nn.ModuleList([ConvPosEnc(dim=in_dim, k=3), ConvPosEnc(dim=in_dim, k=3)]) for _ in range(layer_num))
        self.norm1 = nn.ModuleList(LayerNormProxy(in_dim) for _ in range(layer_num))
        self.norm2 = nn.ModuleList(nn.LayerNorm(in_dim) for _ in range(layer_num))
        self.qkv = nn.ModuleList(nn.Conv2d(in_dim, self.hidden_dim * 3, kernel_size=1, stride=1, padding=0) for _ in range(layer_num))

        mlp_hidden_dim = int(in_dim * mlp_ratio)
        self.mlp = nn.ModuleList(Mlp(in_features=in_dim, hidden_features=mlp_hidden_dim) for _ in range(layer_num))

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.ModuleList(nn.Conv2d(self.hidden_dim, in_dim, kernel_size=1, stride=1, padding=0) for _ in range(layer_num))
        self.pos_embed = CrossLayerPosEmbedding3D(num_heads=num_heads, window_size=self.window_sizes, spatial=True)

    def forward(self, x_list, extra=None):
        WmH, WmW = x_list[-1].shape[-2:]
        shortcut_list = []
        q_list, k_list, v_list = [], [], []

        for i, x in enumerate(x_list):
            B, C, H, W = x.shape
            ws_i, stride_i = self.window_sizes[i], self.stride_list[i]
            pad_i = (
                math.ceil((stride_i * (WmH - 1.0) - H + ws_i) / 2.0),
                math.ceil((stride_i * (WmW - 1.0) - W + ws_i) / 2.0),
            )
            self.padding_list[i] = pad_i
            self.shape_list[i] = [H, W]

            x = self.cpe[i][0](x)
            shortcut_list.append(x)
            qkv = self.qkv[i](x)
            qkv_windows = overlaped_window_partition(qkv, ws_i, stride=stride_i, pad=pad_i)
            qkv_windows = qkv_windows.reshape(B, WmH * WmW, ws_i * ws_i, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
            q_windows, k_windows, v_windows = qkv_windows[0], qkv_windows[1], qkv_windows[2]
            q_list.append(q_windows)
            k_list.append(k_windows)
            v_list.append(v_windows)

        q_stack = torch.cat(q_list, dim=-2)
        k_stack = torch.cat(k_list, dim=-2)
        v_stack = torch.cat(v_list, dim=-2)

        attn = F.normalize(q_stack, dim=-1) @ F.normalize(k_stack, dim=-1).transpose(-1, -2)
        attn = attn + self.pos_embed()
        attn = self.softmax(attn)

        out = attn.to(v_stack.dtype) @ v_stack
        out = out.permute(0, 2, 3, 1, 4).reshape(B, WmH * WmW, self.token_num, self.hidden_dim)

        out_split = out.split(self.token_num_per_layer, dim=-2)
        out_list = []
        for i, out_i in enumerate(out_split):
            ws_i, stride_i, pad_i = self.window_sizes[i], self.stride_list[i], self.padding_list[i]
            H, W = self.shape_list[i]
            out_i = overlaped_window_reverse(out_i, H, W, ws_i, stride_i, pad_i)
            out_i = shortcut_list[i] + self.norm1[i](self.proj[i](out_i))
            out_i = self.cpe[i][1](out_i)
            out_i = out_i.permute(0, 2, 3, 1)
            out_i = out_i + self.mlp[i](self.norm2[i](out_i))
            out_i = out_i.permute(0, 3, 1, 2)
            out_list.append(out_i)
        return out_list


class CrossLayerChannelAttention(nn.Module):
    def __init__(self, in_dim, layer_num=3, alpha=1, num_heads=4, mlp_ratio=2, reduction=4):
        super().__init__()
        assert alpha % 2 != 0, "alpha must be odd"
        self.num_heads = num_heads
        self.reduction = reduction
        self.hidden_dim = in_dim // reduction
        self.in_dim = in_dim
        self.window_sizes = [(4**i + alpha) if i != 0 else (4**i + alpha - 1) for i in range(layer_num)][::-1]
        self.token_num_per_layer = [i for i in self.window_sizes]
        self.token_num = sum(self.token_num_per_layer)

        self.stride_list = [(4**i) for i in range(layer_num)][::-1]
        self.padding_list = [0 for _ in self.window_sizes]
        self.shape_list = [[0, 0] for _ in range(layer_num)]
        self.unshuffle_factor = [(2**i) for i in range(layer_num)][::-1]

        self.cpe = nn.ModuleList(nn.ModuleList([ConvPosEnc(dim=in_dim, k=3), ConvPosEnc(dim=in_dim, k=3)]) for _ in range(layer_num))
        self.norm1 = nn.ModuleList(LayerNormProxy(in_dim) for _ in range(layer_num))
        self.norm2 = nn.ModuleList(nn.LayerNorm(in_dim) for _ in range(layer_num))

        self.qkv = nn.ModuleList(nn.Conv2d(in_dim, self.hidden_dim * 3, kernel_size=1, stride=1, padding=0) for _ in range(layer_num))

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.ModuleList(nn.Conv2d(self.hidden_dim, in_dim, kernel_size=1, stride=1, padding=0) for _ in range(layer_num))

        mlp_hidden_dim = int(in_dim * mlp_ratio)
        self.mlp = nn.ModuleList(Mlp(in_features=in_dim, hidden_features=mlp_hidden_dim) for _ in range(layer_num))

        self.pos_embed = CrossLayerPosEmbedding3D(num_heads=num_heads, window_size=self.window_sizes, spatial=False)

    def forward(self, x_list, extra=None):
        shortcut_list, reverse_shape = [], []
        q_list, k_list, v_list = [], [], []
        for i, x in enumerate(x_list):
            B, C, H, W = x.shape
            self.shape_list[i] = [H, W]
            ws_i, stride_i = self.window_sizes[i], self.stride_list[i]
            pad_i = math.ceil((stride_i * (self.hidden_dim - 1.0) - (self.unshuffle_factor[i]) ** 2 * self.hidden_dim + ws_i) / 2.0)
            self.padding_list[i] = pad_i
            x = self.cpe[i][0](x)
            shortcut_list.append(x)

            qkv = self.qkv[i](x)
            qkv = F.pixel_unshuffle(qkv, downscale_factor=self.unshuffle_factor[i])
            reverse_shape.append(qkv.size(1) // 3)

            qkv_window = einops.rearrange(qkv, "b c h w -> b (h w) c ()")
            qkv_window = overlaped_channel_partition(qkv_window, ws_i, stride=stride_i, pad=pad_i)
            qkv_window = einops.rearrange(qkv_window, "b hw wsm (n nh c) -> n b nh c wsm hw", n=3, nh=self.num_heads)
            q_windows, k_windows, v_windows = qkv_window[0], qkv_window[1], qkv_window[2]
            q_list.append(q_windows)
            k_list.append(k_windows)
            v_list.append(v_windows)

        q_stack = torch.cat(q_list, dim=-2)
        k_stack = torch.cat(k_list, dim=-2)
        v_stack = torch.cat(v_list, dim=-2)

        attn = F.normalize(q_stack, dim=-1) @ F.normalize(k_stack, dim=-1).transpose(-2, -1)
        attn = attn + self.pos_embed()
        attn = self.softmax(attn)

        out = attn.to(v_stack.dtype) @ v_stack
        out = einops.rearrange(out, "b nh c ws hw -> b (nh c) ws hw")

        out_split = out.split(self.token_num_per_layer, dim=-2)
        out_list = []
        for i, out_i in enumerate(out_split):
            ws_i, stride_i, pad_i = self.window_sizes[i], self.stride_list[i], self.padding_list[i]
            out_i = overlaped_channel_reverse(out_i, ws_i, stride_i, pad_i, outC=reverse_shape[i])
            out_i = out_i.permute(0, 2, 1, 3).reshape(B, -1, self.shape_list[-1][0], self.shape_list[-1][1])
            out_i = F.pixel_shuffle(out_i, upscale_factor=self.unshuffle_factor[i])
            out_i = shortcut_list[i] + self.norm1[i](self.proj[i](out_i))
            out_i = self.cpe[i][1](out_i)
            out_i = out_i.permute(0, 2, 3, 1)
            out_i = out_i + self.mlp[i](self.norm2[i](out_i))
            out_i = out_i.permute(0, 3, 1, 2)
            out_list.append(out_i)
        return out_list


# ---------------- 频域融合 ----------------


try:
    from mmcv.ops.carafe import carafe
except Exception:
    carafe = None


def xavier_init(module, distribution="uniform"):
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.xavier_normal_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=True):
    if warning and size is not None and align_corners:
        input_h, input_w = tuple(int(x) for x in input.shape[2:])
        output_h, output_w = tuple(int(x) for x in size)
        if output_h > input_h or output_w > input_w:
            if ((output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1) and
                (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1)):
                warnings.warn(
                    f"When align_corners={align_corners}, "
                    "the output would more aligned if "
                    f"input size {(input_h, input_w)} is `x+1` and "
                    f"out size {(output_h, output_w)} is `nx+1`"
                )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def hamming2D(M, N):
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    return np.outer(hamming_x, hamming_y)


class FreqFusion(nn.Module):
    # TPAMI 2024 Frequency-aware Feature Fusion for Dense Image Prediction
    def __init__(
        self,
        channels,
        scale_factor=1,
        lowpass_kernel=5,
        highpass_kernel=3,
        up_group=1,
        encoder_kernel=3,
        encoder_dilation=1,
        compressed_channels=64,
        align_corners=False,
        upsample_mode="nearest",
        feature_resample=False,
        feature_resample_group=4,
        comp_feat_upsample=True,
        use_high_pass=True,
        use_low_pass=True,
        hr_residual=True,
        semi_conv=True,
        hamming_window=True,
        feature_resample_norm=True,
        **kwargs,
    ):
        super().__init__()
        hr_channels, lr_channels = channels
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = (hr_channels + lr_channels) // 8
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1,
        )

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.feature_resample = feature_resample
        self.comp_feat_upsample = comp_feat_upsample
        if self.feature_resample:
            self.dysampler = LocalSimGuidedSampler(
                in_channels=compressed_channels,
                scale=2,
                style="lp",
                groups=feature_resample_group,
                use_direct_scale=True,
                kernel_size=encoder_kernel,
                norm=feature_resample_norm,
            )
        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1,
            )
        self.hamming_window = hamming_window
        lowpass_pad = 0
        highpass_pad = 0
        if self.hamming_window:
            self.register_buffer(
                "hamming_lowpass",
                torch.FloatTensor(hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad))[None, None,],
            )
            self.register_buffer(
                "hamming_highpass",
                torch.FloatTensor(hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,],
            )
        else:
            self.register_buffer("hamming_lowpass", torch.FloatTensor([1.0]))
            self.register_buffer("hamming_highpass", torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, x, use_checkpoint=False):
        hr_feat, lr_feat = x
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        if carafe is None:
            raise ImportError("FreqFusion requires mmcv.ops.carafe in current implementation.")
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        if self.semi_conv:
            if self.comp_feat_upsample:
                if self.use_high_pass:
                    mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                    mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel, hamming=self.hamming_highpass)
                    compressed_hr_feat = (
                        compressed_hr_feat
                        + compressed_hr_feat
                        - carafe(compressed_hr_feat, mask_hr_init.to(compressed_hr_feat.dtype), self.highpass_kernel, self.up_group, 1)
                    )

                    mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
                    mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, hamming=self.hamming_lowpass)

                    mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
                    mask_lr_lr_feat = F.interpolate(
                        carafe(mask_lr_lr_feat_lr, mask_lr_init.to(compressed_hr_feat.dtype), self.lowpass_kernel, self.up_group, 2),
                        size=compressed_hr_feat.shape[-2:],
                        mode="nearest",
                    )
                    mask_lr = mask_lr_hr_feat + mask_lr_lr_feat

                    mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
                    mask_hr_lr_feat = F.interpolate(
                        carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init.to(compressed_hr_feat.dtype), self.lowpass_kernel, self.up_group, 2),
                        size=compressed_hr_feat.shape[-2:],
                        mode="nearest",
                    )
                    mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
                else:
                    raise NotImplementedError
            else:
                mask_lr = self.content_encoder(compressed_hr_feat) + F.interpolate(
                    self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode="nearest"
                )
                if self.use_high_pass:
                    mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(
                        self.content_encoder2(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode="nearest"
                    )
        else:
            compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[-2:], mode="nearest") + compressed_hr_feat
            mask_lr = self.content_encoder(compressed_x)
            if self.use_high_pass:
                mask_hr = self.content_encoder2(compressed_x)

        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
        if self.semi_conv:
            lr_feat = carafe(lr_feat, mask_lr.to(lr_feat.dtype), self.lowpass_kernel, self.up_group, 2)
        else:
            lr_feat = resize(
                input=lr_feat,
                size=hr_feat.shape[2:],
                mode=self.upsample_mode,
                align_corners=None if self.upsample_mode == "nearest" else self.align_corners,
            )
            lr_feat = carafe(lr_feat, mask_lr.to(lr_feat.dtype), self.lowpass_kernel, self.up_group, 1)

        if self.use_high_pass:
            mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
            if self.hr_residual:
                hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr.to(hr_feat.dtype), self.highpass_kernel, self.up_group, 1)
                hr_feat = hr_feat_hf + hr_feat
            else:
                hr_feat = hr_feat_hf

        if self.feature_resample:
            lr_feat = self.dysampler(hr_x=compressed_hr_feat, lr_x=compressed_lr_feat, feat2sample=lr_feat)
        return hr_feat + lr_feat


def compute_similarity(input_tensor, k=3, dilation=1, sim="cos"):
    B, C, H, W = input_tensor.shape
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)
    unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)

    if sim == "cos":
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    elif sim == "dot":
        similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError

    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)
    similarity = similarity.view(B, k * k - 1, H, W)
    return similarity


class LocalSimGuidedSampler(nn.Module):
    """offset generator in FreqFusion"""

    def __init__(
        self,
        in_channels,
        scale=2,
        style="lp",
        groups=4,
        use_direct_scale=True,
        kernel_size=1,
        local_window=3,
        sim_type="cos",
        norm=True,
        direction_feat="sim_concat",
    ):
        super().__init__()
        assert scale == 2
        assert style == "lp"

        self.scale = scale
        self.style = style
        self.groups = groups
        self.local_window = local_window
        self.sim_type = sim_type
        self.direction_feat = direction_feat

        if style == "pl":
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == "pl":
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2
        if self.direction_feat == "sim":
            self.offset = nn.Conv2d(local_window ** 2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        elif self.direction_feat == "sim_concat":
            self.offset = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        else:
            raise NotImplementedError
        normal_init(self.offset, std=0.001)
        if use_direct_scale:
            if self.direction_feat == "sim":
                self.direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            elif self.direction_feat == "sim_concat":
                self.direct_scale = nn.Conv2d(
                    in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
                )
            else:
                raise NotImplementedError
            constant_init(self.direct_scale, val=0.0)

        out_channels = 2 * groups
        if self.direction_feat == "sim":
            self.hr_offset = nn.Conv2d(local_window ** 2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        elif self.direction_feat == "sim_concat":
            self.hr_offset = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        else:
            raise NotImplementedError
        normal_init(self.hr_offset, std=0.001)

        if use_direct_scale:
            if self.direction_feat == "sim":
                self.hr_direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            elif self.direction_feat == "sim_concat":
                self.hr_direct_scale = nn.Conv2d(
                    in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
                )
            else:
                raise NotImplementedError
            constant_init(self.hr_direct_scale, val=0.0)

        self.norm = norm
        if self.norm:
            self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels)
            self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels)
        else:
            self.norm_hr = nn.Identity()
            self.norm_lr = nn.Identity()
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing="ij")).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset, scale=None):
        if scale is None:
            scale = self.scale
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(W, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij")).transpose(1, 2).unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(B, -1, H, W), scale)
            .view(B, 2, -1, scale * H, scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(B * self.groups, -1, x.size(-2), x.size(-1)),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, scale * H, scale * W)

    def forward(self, hr_x, lr_x, feat2sample):
        hr_x = self.norm_hr(hr_x)
        lr_x = self.norm_lr(lr_x)

        if self.direction_feat == "sim":
            hr_sim = compute_similarity(hr_x, self.local_window, dilation=2, sim="cos")
            lr_sim = compute_similarity(lr_x, self.local_window, dilation=2, sim="cos")
        elif self.direction_feat == "sim_concat":
            hr_sim = torch.cat([hr_x, compute_similarity(hr_x, self.local_window, dilation=2, sim="cos")], dim=1)
            lr_sim = torch.cat([lr_x, compute_similarity(lr_x, self.local_window, dilation=2, sim="cos")], dim=1)
            hr_x, lr_x = hr_sim, lr_sim
        else:
            raise NotImplementedError

        offset = self.get_offset_lp(hr_x, lr_x, hr_sim, lr_sim)
        return self.sample(feat2sample, offset)

    def get_offset_lp(self, hr_x, lr_x, hr_sim, lr_sim):
        if hasattr(self, "direct_scale"):
            offset = (
                self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)
            ) * (
                self.direct_scale(lr_x) + F.pixel_unshuffle(self.hr_direct_scale(hr_x), self.scale)
            ).sigmoid() + self.init_pos
        else:
            offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * 0.25 + self.init_pos
        return offset

    def get_offset(self, hr_x, lr_x):
        if self.style == "pl":
            raise NotImplementedError
        return self.get_offset_lp(hr_x, lr_x)


# ---------------- OREPA / RepPAN ----------------


def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std


def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [W_pixels_to_pad, W_pixels_to_pad, H_pixels_to_pad, H_pixels_to_pad])


class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class OREPA(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        act=True,
        internal_channels_1x1_3x3=None,
        deploy=False,
        single_init=False,
        weight_only=False,
        init_hyper_para=1.0,
        init_hyper_gamma=1.0,
    ):
        super().__init__()
        self.deploy = deploy
        self.nonlinear = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.weight_only = weight_only
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        padding = autopad(kernel_size, padding, dilation)
        self.padding = padding
        self.dilation = dilation

        if deploy:
            self.orepa_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.branch_counter = 0
            self.weight_orepa_origin = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), kernel_size, kernel_size))
            nn.init.kaiming_uniform_(self.weight_orepa_origin, a=math.sqrt(0.0))
            self.branch_counter += 1

            self.weight_orepa_avg_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            self.weight_orepa_pfir_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_orepa_avg_conv, a=0.0)
            nn.init.kaiming_uniform_(self.weight_orepa_pfir_conv, a=0.0)
            self.register_buffer("weight_orepa_avg_avg", torch.ones(kernel_size, kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 2

            self.weight_orepa_1x1 = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_orepa_1x1, a=0.0)
            self.branch_counter += 1

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups <= 4 else 2 * in_channels

            if internal_channels_1x1_3x3 == in_channels:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
                id_value = np.zeros((in_channels, int(in_channels / self.groups), 1, 1))
                for i in range(in_channels):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer("id_tensor", id_tensor)
            else:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1)
                )
                id_value = np.zeros((internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                for i in range(internal_channels_1x1_3x3):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer("id_tensor", id_tensor)

            self.weight_orepa_1x1_kxk_conv2 = nn.Parameter(
                torch.Tensor(out_channels, int(internal_channels_1x1_3x3 / self.groups), kernel_size, kernel_size)
            )
            nn.init.kaiming_uniform_(self.weight_orepa_1x1_kxk_conv2, a=math.sqrt(0.0))
            self.branch_counter += 1

            expand_ratio = 8
            self.weight_orepa_gconv_dw = nn.Parameter(torch.Tensor(in_channels * expand_ratio, 1, kernel_size, kernel_size))
            self.weight_orepa_gconv_pw = nn.Parameter(torch.Tensor(out_channels, int(in_channels * expand_ratio / self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_orepa_gconv_dw, a=math.sqrt(0.0))
            nn.init.kaiming_uniform_(self.weight_orepa_gconv_pw, a=math.sqrt(0.0))
            self.branch_counter += 1

            self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            if weight_only is False:
                self.bn = nn.BatchNorm2d(self.out_channels)

            self.fre_init()

            nn.init.constant_(self.vector[0, :], 0.25 * math.sqrt(init_hyper_gamma))
            nn.init.constant_(self.vector[1, :], 0.25 * math.sqrt(init_hyper_gamma))
            nn.init.constant_(self.vector[2, :], 0.0 * math.sqrt(init_hyper_gamma))
            nn.init.constant_(self.vector[3, :], 0.5 * math.sqrt(init_hyper_gamma))
            nn.init.constant_(self.vector[4, :], 1.0 * math.sqrt(init_hyper_gamma))
            nn.init.constant_(self.vector[5, :], 0.5 * math.sqrt(init_hyper_gamma))

            self.weight_orepa_1x1.data = self.weight_orepa_1x1.mul(init_hyper_para)
            self.weight_orepa_origin.data = self.weight_orepa_origin.mul(init_hyper_para)
            self.weight_orepa_1x1_kxk_conv2.data = self.weight_orepa_1x1_kxk_conv2.mul(init_hyper_para)
            self.weight_orepa_avg_conv.data = self.weight_orepa_avg_conv.mul(init_hyper_para)
            self.weight_orepa_pfir_conv.data = self.weight_orepa_pfir_conv.mul(init_hyper_para)
            self.weight_orepa_gconv_dw.data = self.weight_orepa_gconv_dw.mul(math.sqrt(init_hyper_para))
            self.weight_orepa_gconv_pw.data = self.weight_orepa_gconv_pw.mul(math.sqrt(init_hyper_para))

            if single_init:
                self.single_init()

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) * (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) * (i + 1 - half_fg) / 3)
        self.register_buffer("weight_orepa_prior", prior_tensor)

    def weight_gen(self):
        weight_orepa_origin = torch.einsum("oihw,o->oihw", self.weight_orepa_origin, self.vector[0, :])
        weight_orepa_avg = torch.einsum(
            "oi,hw->oihw", self.weight_orepa_avg_conv.squeeze(3).squeeze(2), self.weight_orepa_avg_avg
        )
        weight_orepa_avg = torch.einsum("oihw,o->oihw", weight_orepa_avg, self.vector[1, :])
        weight_orepa_pfir = torch.einsum(
            "oi,ohw->oihw", self.weight_orepa_pfir_conv.squeeze(3).squeeze(2), self.weight_orepa_prior
        )
        weight_orepa_pfir = torch.einsum("oihw,o->oihw", weight_orepa_pfir, self.vector[2, :])

        if hasattr(self, "weight_orepa_1x1_kxk_idconv1"):
            weight_orepa_1x1_kxk_conv1 = (self.weight_orepa_1x1_kxk_idconv1 + self.id_tensor).squeeze(3).squeeze(2)
        else:
            weight_orepa_1x1_kxk_conv1 = self.weight_orepa_1x1_kxk_conv1.squeeze(3).squeeze(2)
        weight_orepa_1x1_kxk_conv2 = self.weight_orepa_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_orepa_1x1_kxk_conv1.size()
            o, tg, h, w = weight_orepa_1x1_kxk_conv2.size()
            weight_orepa_1x1_kxk_conv1 = weight_orepa_1x1_kxk_conv1.view(g, int(t / g), ig)
            weight_orepa_1x1_kxk_conv2 = weight_orepa_1x1_kxk_conv2.view(g, int(o / g), tg, h, w)
            weight_orepa_1x1_kxk = torch.einsum("gti,gothw->goihw", weight_orepa_1x1_kxk_conv1, weight_orepa_1x1_kxk_conv2).reshape(o, ig, h, w)
        else:
            weight_orepa_1x1_kxk = torch.einsum("ti,othw->oihw", weight_orepa_1x1_kxk_conv1, weight_orepa_1x1_kxk_conv2)
        weight_orepa_1x1_kxk = torch.einsum("oihw,o->oihw", weight_orepa_1x1_kxk, self.vector[3, :])

        weight_orepa_1x1 = transVI_multiscale(self.weight_orepa_1x1, self.kernel_size)
        weight_orepa_1x1 = torch.einsum("oihw,o->oihw", weight_orepa_1x1, self.vector[4, :])

        weight_orepa_gconv = torch.einsum("oihw,o->oihw", self.weight_orepa_gconv_pw, self.vector[5, :])
        weight_orepa_gconv_dw = transVI_multiscale(self.weight_orepa_gconv_dw, self.kernel_size)
        weight_orepa_gconv = torch.einsum("oihw,o->oihw", weight_orepa_gconv, self.vector[5, :])
        weight_orepa_gconv = torch.einsum("oihw,ohw->oihw", weight_orepa_gconv, weight_orepa_gconv_dw.squeeze(1))

        weight = weight_orepa_origin + weight_orepa_avg + weight_orepa_pfir + weight_orepa_1x1_kxk + weight_orepa_1x1 + weight_orepa_gconv
        return weight

    def forward(self, x):
        if hasattr(self, "orepa_reparam"):
            return self.nonlinear(self.orepa_reparam(x))
        weight = self.weight_gen()
        out = F.conv2d(x, weight, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
        if hasattr(self, "bn"):
            out = self.bn(out)
        return self.nonlinear(out)


class RepConvN(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)


class RepNBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class OREPANBottleneck(RepNBottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = OREPA(c1, c_, k[0], 1)


class RepNCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class OREPANCSP(RepNCSP):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(OREPANBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class OREPANCSPELAN4(RepNCSPELAN4):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__(c1, c2, c3, c4, c5)
        self.cv2 = nn.Sequential(OREPANCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(OREPANCSP(c4, c4, c5), Conv(c4, c4, 3, 1))


# ---------------- DAMO-YOLO GFPN ----------------


class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self, ch_in, ch_hidden_ratio, ch_out, shortcut=True):
        super().__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = Conv(ch_hidden, ch_out, 3, s=1)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, s=1)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        return x + y if self.shortcut else y


class SPP_DAMO(nn.Module):
    def __init__(self, ch_in, ch_out, k, pool_size):
        super().__init__()
        self.pool = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
                for size in pool_size
            ]
        )
        self.conv = Conv(ch_in, ch_out, k)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)
        return self.conv(y)


class CSPStage(nn.Module):
    def __init__(self, ch_in, ch_out, n, block_fn="BasicBlock_3x3_Reverse", ch_hidden_ratio=1.0, spp=False):
        super().__init__()
        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = Conv(ch_in, ch_first, 1)
        self.conv2 = Conv(ch_in, ch_mid, 1)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == "BasicBlock_3x3_Reverse":
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in, ch_hidden_ratio, ch_mid, shortcut=True),
                )
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module("spp", SPP_DAMO(ch_mid * 4, ch_mid, 1, [5, 9, 13]))
            next_ch_in = ch_mid
        self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        return self.conv3(y)


# ---------------- EfficientRepBiPAN / RepPAN ----------------


class Transpose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=True)

    def forward(self, x):
        return self.upsample_transpose(x)


class BiFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = Conv(in_channels[1], out_channels, 1, 1)
        self.cv2 = Conv(in_channels[2], out_channels, 1, 1)
        self.cv3 = Conv(out_channels * 3, out_channels, 1, 1)
        self.upsample = Transpose(in_channels=out_channels, out_channels=out_channels)
        self.downsample = Conv(out_channels, out_channels, 3, 2)

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))


# ---------------- Re-Calibration FPN ----------------


def Upsample_bilinear(x, size, align_corners=False):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


class SBA(nn.Module):
    def __init__(self, inc, input_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.d_in1 = Conv(input_dim // 2, input_dim // 2, 1)
        self.d_in2 = Conv(input_dim // 2, input_dim // 2, 1)
        self.conv = Conv(input_dim, input_dim, 3)
        self.fc1 = nn.Conv2d(inc[1], input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], input_dim // 2, kernel_size=1, bias=False)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        H_feature, L_feature = x
        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)
        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)
        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)
        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample_bilinear(
            g_H_feature * H_feature, size=L_feature.size()[2:], align_corners=False
        )
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample_bilinear(
            g_L_feature * L_feature, size=H_feature.size()[2:], align_corners=False
        )
        H_feature = Upsample_bilinear(H_feature, size=L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out


class LayerNormCF(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class CABAttn(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        if dim % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = einops.rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = einops.rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = einops.rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = einops.rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class IEL_LCA(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias
        )
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.tanh(self.dwconv1(x1)) + x1
        x2 = self.tanh(self.dwconv2(x2)) + x2
        return self.project_out(x1 * x2)


class CrossAttentionBlock(nn.Module):
    """CAB-style cross-attention fusion block used by rtdetr-CAB variants."""

    def __init__(self, in_dim, out_dim, num_heads=8, bias=False):
        super().__init__()
        self.norm = LayerNormCF(out_dim)
        self.gdfn = IEL_LCA(out_dim)
        self.ffn = CABAttn(out_dim, num_heads=num_heads, bias=bias)
        self.conv1x1 = nn.ModuleList(Conv(c, out_dim, 1) if c != out_dim else nn.Identity() for c in in_dim)

    def forward(self, inputs):
        x, y = inputs
        x = self.conv1x1[0](x)
        y = self.conv1x1[1](y)
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x


class AdaptiveCombiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, p, i):
        batch_size, channel, w, h = p.shape
        d = self.d.expand(batch_size, channel, w, h)
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i


class DPCF(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ac = AdaptiveCombiner()
        self.tail_conv = Conv(in_features[1], out_features)
        self.conv1x1 = Conv(in_features[0], in_features[1], 1) if in_features[0] != in_features[1] else nn.Identity()

    def forward(self, inputs):
        x_low, x_high = inputs
        x_low = self.conv1x1(x_low)
        image_size = x_high.size(2)

        if x_high is not None:
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low is not None:
            x_low = F.interpolate(x_low, size=[image_size, image_size], mode="bilinear", align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1)

        x0 = self.ac(x_low[0], x_high[0])
        x1 = self.ac(x_low[1], x_high[1])
        x2 = self.ac(x_low[2], x_high[2])
        x3 = self.ac(x_low[3], x_high[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x


class GDSAFusion(nn.Module):
    """Lightweight compatibility implementation of GDSAFusion for neck feature fusion."""

    def __init__(self, c1, c2):
        super().__init__()
        self.align2 = Conv(c2, c1, 1, 1, act=False) if c1 != c2 else nn.Identity()
        self.mix = Conv(c1 * 2, c1, 1, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1, x2 = x
        if x2.shape[2:] != x1.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode="nearest")
        x2 = self.align2(x2)
        fused = self.mix(torch.cat([x1, x2], dim=1))
        w = self.gate(fused)
        return x1 + w * fused


class FuseModule(nn.Module):
    def __init__(self, c_in, channel_adjust=False):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        in_ch = 4 * c_in if channel_adjust else 3 * c_in
        self.conv_out = Conv(in_ch, c_in, 1)

    def forward(self, x):
        x1_ds = F.adaptive_avg_pool2d(x[0], x[1].shape[2:])
        x3_up = F.interpolate(x[2], size=x[1].shape[2:], mode="nearest")
        x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
        return self.conv_out(x_cat)


class HyperACE(nn.Module):
    """Lightweight HyperACE-compatible module with the same YAML interface."""

    def __init__(
        self,
        in_features,
        out_features,
        n=1,
        num_hyperedges=8,
        dsc3k=True,
        shortcut=False,
        e1=0.5,
        e2=1.0,
        context="both",
        channel_adjust=False,
    ):
        super().__init__()
        _ = (num_hyperedges, dsc3k, shortcut, e2, context)  # reserved for interface compatibility
        c_in = int(in_features[1])
        self.align = nn.ModuleList(
            Conv(c, c_in, 1) if c != c_in else nn.Identity() for c in in_features
        )
        self.fuse = FuseModule(c_in, channel_adjust=channel_adjust)
        self.c = int(out_features * e1)
        self.cv1 = Conv(c_in, 3 * self.c, 1, 1)
        self.low = nn.Sequential(*(Conv(self.c, self.c, 3, 1) for _ in range(max(1, int(n)))))
        self.branch1 = Conv(self.c, self.c, 3, 1)
        self.branch2 = Conv(self.c, self.c, 3, 1)
        self.cv2 = Conv(4 * self.c, out_features, 1, 1)

    def forward(self, x):
        x = [self.align[i](x[i]) for i in range(3)]
        x = self.fuse(x)
        y0, y1, y2 = self.cv1(x).chunk(3, 1)
        out1 = self.branch1(y1)
        out2 = self.branch2(y1)
        low = self.low(y2)
        return self.cv2(torch.cat([y0, out1, low, out2], 1))


class FullPAD_Tunnel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x[0] + self.gate * x[1]


# ---------------- Efficient Multi-Branch & Scale FPN ----------------


class EUCB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.up_dwc = nn.Sequential(nn.Upsample(scale_factor=2), Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels, s=stride))
        self.pwc = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class Shift_channel_mix(nn.Module):
    def __init__(self, shift_size):
        super().__init__()
        self.shift_size = shift_size

    def forward(self, x):
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        x1 = torch.roll(x1, self.shift_size, dims=2)
        x2 = torch.roll(x2, -self.shift_size, dims=2)
        x3 = torch.roll(x3, self.shift_size, dims=3)
        x4 = torch.roll(x4, -self.shift_size, dims=3)
        return torch.cat([x1, x2, x3, x4], 1)


class EUCB_SC(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.up_dwc = nn.Sequential(nn.Upsample(scale_factor=2), Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels, s=stride))
        self.pwc = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.shift_channel_mix = Shift_channel_mix(1)

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        x = self.shift_channel_mix(x)
        return x


class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, dw_parallel=True):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([nn.Sequential(Conv(self.in_channels, self.in_channels, kernel_size, s=stride, g=self.in_channels)) for kernel_size in self.kernel_sizes])

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5], stride=1, expansion_factor=2, dw_parallel=True, add=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.n_scales = len(self.kernel_sizes)
        assert self.stride in [1, 2]
        self.use_skip_connection = True if self.stride == 1 else False
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(Conv(self.in_channels, self.ex_channels, 1))
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, dw_parallel=self.dw_parallel)
        self.combined_channels = self.ex_channels if self.add else self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(Conv(self.combined_channels, self.out_channels, 1, act=False))
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = self.channel_shuffle(dout, math.gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class MSCB_SC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5], stride=1, expansion_factor=2, dw_parallel=True, add=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.n_scales = len(self.kernel_sizes)
        assert self.stride in [1, 2]
        self.use_skip_connection = True if self.stride == 1 else False
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(Conv(self.in_channels, self.ex_channels, 1))
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, dw_parallel=self.dw_parallel)
        self.combined_channels = self.ex_channels if self.add else self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(Conv(self.combined_channels, self.out_channels, 1, act=False))
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.shift_channel_mix = Shift_channel_mix(1)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = self.channel_shuffle(dout, math.gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        x = self.shift_channel_mix(x)
        return x


class CSP_MSCB(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=[1, 3, 5], shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MSCB(self.c, self.c, kernel_sizes=kernel_sizes) for _ in range(n))


class CSP_MSCB_SC(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=[1, 3, 5], shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MSCB_SC(self.c, self.c, kernel_sizes=kernel_sizes) for _ in range(n))
