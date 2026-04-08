# 文件: ultralytics/nn/modules/fusion/sigmafusion.py
# 来源: Sigma (CVPR 2024) — Siamese Mamba Network for Multi-Modal Semantic Segmentation
# 核心: 跨模态选择性门控交互 (Cross-Modal Selective Gating)
# 二次创新:
#   1. FFT 频率解耦路径 —— 将高频(细节)与低频(能量)分流处理后再门控融合
#   2. Spatial-Gate 空间门控 —— 生成像素级共用空间掩码，实现空间选择性融合
#   3. 输出 BN 稳定 —— 防止 AMP fp16 数值溢出

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmaFusionBlock(nn.Module):
    """
    Sigma 跨模态选择性融合模块（二次创新增强版）。

    设计三阶段：
      Stage 1 (频率解耦): 对两路特征分别做 FFT，分离高频/低频分量，
            分别通过独立通道门控加权，保留模态互补频率信息。
      Stage 2 (跨模态选择性门控): 用红外门控作用于可见光，用可见光门控作用于红外，
            模拟 Sigma 论文的 Cross-Selective Scan 权重互换机制。
      Stage 3 (像素级空间门控): 利用两路特征差异生成共用空间掩码，
            像素级过滤噪声并聚合，DWConv 增强空间结构感知能力。

    输入: List[Tensor] -> [x_rgb, x_ir]，shape 相同 [B, C, H, W]
    输出: Tensor，shape [B, C, H, W]（通道数不变，无膨胀）

    Args:
        c1 (int): 两路输入通道数（应相同）。
    """

    def __init__(self, c1):
        super().__init__()
        self.c = c1
        mid = max(1, c1 // 4)

        # ---- Stage 1: 频率解耦通道门控 ----
        # 低频分支（全局能量感知）
        self.freq_gate_low = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c1, 1, bias=False),
            nn.Sigmoid(),
        )
        # 高频分支（细节边缘感知）
        self.freq_gate_high = nn.Sequential(
            nn.Conv2d(c1, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c1, 1, bias=False),
            nn.Sigmoid(),
        )

        # ---- Stage 2: 跨模态选择性通道门控 (Sigma Cross-Modal Gating) ----
        # 红外引导可见光的权重
        self.gate_ir2rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c1, 1, bias=False),
            nn.Sigmoid(),
        )
        # 可见光引导红外的权重
        self.gate_rgb2ir = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c1, 1, bias=False),
            nn.Sigmoid(),
        )

        # ---- Stage 3: 空间门控 (Spatial-Gate) ----
        # 利用两路模态差异图生成像素级空间掩码
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # 深度可分离卷积：聚合空间结构，增强小目标感知
        self.fuse_dw = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # 输出 BN 稳定（AMP fp16 防溢出）
        self.bn_out = nn.BatchNorm2d(c1)

    def _fft_decompose(self, x):
        """
        FFT 频率分解：将特征图分离为低频（全局）和高频（细节）分量。
        低频 = 对振幅图做 GAP 重构；高频 = 原图 - 低频。
        返回: (low_freq, high_freq)，shape 均为 [B, C, H, W]
        """
        # rfft2: [B, C, H, W/2+1] 复数
        fft_feat = torch.fft.rfft2(x.float(), norm='ortho')
        # 振幅谱
        amp = torch.abs(fft_feat)
        # 低频：GAP 全局能量（标量，广播到全频率）
        low_amp = amp.mean(dim=(-2, -1), keepdim=True).expand_as(amp)
        high_amp = amp - low_amp  # 高频 = 残差振幅
        # 用相位重构低/高频空间特征
        phase = torch.angle(fft_feat)
        low_fft = torch.polar(low_amp.clamp(min=0), phase)
        high_fft = torch.polar(high_amp.clamp(min=0), phase)
        low_feat = torch.fft.irfft2(low_fft, s=x.shape[-2:], norm='ortho').to(x.dtype)
        high_feat = torch.fft.irfft2(high_fft, s=x.shape[-2:], norm='ortho').to(x.dtype)
        return low_feat, high_feat

    def forward(self, x):
        """
        Args:
            x: List[Tensor]，[x_rgb, x_ir]，shape 需相同 [B, C, H, W]。
        Returns:
            Tensor: 融合后特征，shape [B, C, H, W]。
        """
        x_rgb, x_ir = x[0], x[1]

        # ---- Stage 1: FFT 频率解耦增强 ----
        rgb_low, rgb_high = self._fft_decompose(x_rgb)
        ir_low, ir_high = self._fft_decompose(x_ir)

        # 低频：GAP 通道门控（全局能量感知）
        w_low_rgb = self.freq_gate_low(rgb_low)
        w_low_ir = self.freq_gate_low(ir_low)
        # 高频：逐像素通道门控（细节感知）
        w_high_rgb = self.freq_gate_high(rgb_high)
        w_high_ir = self.freq_gate_high(ir_high)

        # 重组：低频 + 高频加权后叠加，保持各自模态完整语义
        f_rgb_freq = rgb_low * w_low_rgb + rgb_high * w_high_rgb
        f_ir_freq = ir_low * w_low_ir + ir_high * w_high_ir

        # 频率增强残差叠加回原始特征
        f_rgb = x_rgb + f_rgb_freq
        f_ir = x_ir + f_ir_freq

        # ---- Stage 2: 跨模态选择性门控 (Sigma 核心) ----
        # 红外门控引导可见光 / 可见光门控引导红外
        w_ir2rgb = self.gate_ir2rgb(f_ir)   # 红外生成权重，作用于可见光
        w_rgb2ir = self.gate_rgb2ir(f_rgb)  # 可见光生成权重，作用于红外

        f_rgb_gated = f_rgb * w_ir2rgb   # 红外引导可见光
        f_ir_gated = f_ir * w_rgb2ir     # 可见光引导红外

        # 跨模态互补相加
        f_cross = f_rgb_gated + f_ir_gated

        # ---- Stage 3: 空间门控 (Spatial-Gate) ----
        # 用两路差异生成像素级掩码，过滤噪声，增强共同显著区域
        diff = torch.abs(f_rgb - f_ir)
        spatial_mask = self.spatial_gate(diff)    # [B, 1, H, W]
        f_spatial = f_cross * spatial_mask         # 像素级选择性保留

        # DWConv 空间结构聚合 + 残差
        out = self.fuse_dw(f_spatial) + f_cross
        return self.bn_out(out)
