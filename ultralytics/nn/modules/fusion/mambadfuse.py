# 文件: ultralytics/nn/modules/fusion/mambadfuse.py
# 来源: MambaDFuse (arXiv:2404.08406)
# 核心: 双阶段融合 —— 浅层通道交换（Channel Exchange）+ 深层多模态门控（M3 Block 简化）
# 注意: 完整 Mamba SSM 需 mamba_ssm 库，此处以高效 DWConv 模拟空间选择性建模能力

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class MambaDFuseBlock(nn.Module):
    """
    MambaDFuse: Dual-phase Multi-modality Fusion Block.

    双阶段设计：
      Stage 1 (Shallow): 通道交换（Channel Exchange），零参数基础交互。
      Stage 2 (Deep):    多模态门控（M3 Block 简化版），利用原始模态特征
                          生成 SE 门控信号，对浅层融合特征加权增强。

    输入: List[Tensor] -> [x_rgb, x_ir]，shape 相同 [B, C, H, W]
    输出: Tensor，shape [B, C, H, W]，通道数保持不变

    Args:
        c1 (int): 第一路（RGB）输入通道数。
        c2 (int): 第二路（IR/X）输入通道数（应与 c1 相同）。
        p  (float): 通道交换比例，默认 0.5。
    """

    def __init__(self, c1, c2, p=0.5):
        super().__init__()
        assert c1 == c2, f"MambaDFuseBlock requires c1==c2, got c1={c1}, c2={c2}"
        self.c = c1
        self.p = p

        # Stage 2: 门控路径（模拟 M3 Block 的选择性机制）
        # 利用两路原始特征各自生成 SE 通道门控
        self.gate_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid(),
        )
        self.gate_ir = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid(),
        )

        # 融合特征增强（类 Mamba 投影层，使用 DWConv 模拟空间选择性建模）
        self.project = Conv(c1, c1, 3, 1)

        # 输出 BN 稳定训练（AMP fp16 下防止数值溢出）
        self.bn_out = nn.BatchNorm2d(c1)

    def forward(self, x):
        """
        Args:
            x: List[Tensor]，[x_rgb, x_ir]，shape 需相同。
        Returns:
            Tensor: 融合后特征，shape [B, C, H, W]。
        """
        x_rgb, x_ir = x[0], x[1]

        # --- Stage 1: Shallow Fusion（Channel Exchange）---
        _, channels, _, _ = x_rgb.shape
        exchange_c = int(channels * self.p)

        # 物理通道交换（无参）
        out_rgb = torch.cat([x_ir[:, :exchange_c], x_rgb[:, exchange_c:]], dim=1)
        out_ir = torch.cat([x_rgb[:, :exchange_c], x_ir[:, exchange_c:]], dim=1)

        # 基础相加，得到浅层融合特征
        f_shallow = out_rgb + out_ir

        # --- Stage 2: Deep Fusion（M3 Block Gating）---
        # 利用原始模态特征生成门控信号（保留原始分布语义）
        g_rgb = self.gate_rgb(x_rgb)
        g_ir = self.gate_ir(x_ir)

        # 深度卷积增强浅层特征（模拟 Mamba 投影）
        f_deep = self.project(f_shallow)

        # 门控加权融合（对应论文公式 10/11）
        f_fused = f_deep * (g_rgb + g_ir)

        return self.bn_out(f_fused)
