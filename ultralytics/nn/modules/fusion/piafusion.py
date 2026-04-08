# 文件: ultralytics/nn/modules/fusion/piafusion.py
# 来源: PIAFusion (Information Fusion 2022)
# 论文: "PIAFusion: A progressive infrared and visible image fusion network
#        based on illumination aware"
# 核心: 光照感知权重 (Illumination-Aware Weighting) +
#       跨模态微分感知融合 (CMDAF, Cross-Modality Differential Aware Fusion)
# 输出: torch.cat([f_vi, f_ir], dim=1) -> 2*C 通道，需后接 Conv[C,1,1] 降维

import torch
import torch.nn as nn


class PIAFusionBlock(nn.Module):
    """
    PIAFusion: 基于光照感知的渐进式跨模态特征融合模块。

    双阶段设计：
      Stage 1 (Illumination Weighting): 从可见光特征预测场景亮度，生成
            动态权重 w_vi / w_ir（softmax 归一化，和为 1）。
      Stage 2 (CMDAF): 利用两模态差值特征引导通道注意力，分别对 RGB/IR
            特征做互补增强（论文 Eq.5）。
      Stage 3 (Weighted Concat): 加权拼接，输出 2*C 通道。

    输入: List[Tensor] -> [x_rgb, x_ir]，shape 相同 [B, C, H, W]
    输出: Tensor，shape [B, 2*C, H, W]（需后接 Conv[C, 1, 1] 降维到 C）

    Args:
        c1 (int): RGB 输入通道数。
        c2 (int): IR/X 输入通道数（应与 c1 相同）。
    """

    def __init__(self, c1, c2):
        super().__init__()
        assert c1 == c2, f"PIAFusionBlock requires c1==c2, got c1={c1}, c2={c2}"
        self.c = c1

        # ---- Stage 1: 光照感知权重预测 ----
        # 输入可见光全局信息，预测 [w_vi, w_ir] 两个 softmax 归一化权重
        self.illum_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # [B, C, 1, 1]
            nn.Conv2d(c1, max(1, c1 // 4), 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(max(1, c1 // 4), 2, 1, bias=False),
            nn.Softmax(dim=1),                 # 输出 [B, 2, 1, 1]
        )

        # ---- Stage 2: CMDAF 通道注意力 ----
        # 差值特征经 GAP 后通过 1x1 Conv + Sigmoid 生成通道注意力图
        self.channel_att = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.Sigmoid(),
        )

        # 用于对 GAP 后的差值向量做通道映射（复用 channel_att 已足够）
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Args:
            x: List[Tensor]，[x_rgb, x_ir]，shape 需相同 [B, C, H, W]。
        Returns:
            Tensor: shape [B, 2*C, H, W]，通道拼接结果。
        """
        x_rgb, x_ir = x[0], x[1]

        # ---- Stage 1: Illumination Weighting ----
        # 以可见光特征预测场景亮度，生成双模态权重
        illum_weights = self.illum_net(x_rgb)        # [B, 2, 1, 1]
        w_vi = illum_weights[:, 0:1, :, :]           # [B, 1, 1, 1]
        w_ir = illum_weights[:, 1:2, :, :]           # [B, 1, 1, 1]

        # ---- Stage 2: CMDAF（差值感知通道注意力）----
        # 对差值特征取 GAP 再通过 channel_att 生成注意力向量
        diff_ir = x_ir - x_rgb                       # 红外相对可见光的差异
        diff_vi = x_rgb - x_ir                       # 可见光相对红外的差异

        att_ir = self.channel_att(self.gap(diff_ir)) # [B, C, 1, 1]
        att_vi = self.channel_att(self.gap(diff_vi)) # [B, C, 1, 1]

        # 互补增强（论文 Eq.5）
        f_ir = x_ir + att_ir * diff_vi               # 红外特征加入可见光差异补偿
        f_vi = x_rgb + att_vi * diff_ir              # 可见光特征加入红外差异补偿

        # ---- Stage 3: Weighted Concatenation ----
        # 光照权重加权后拼接，输出 2*C 通道
        out = torch.cat([f_vi * w_vi, f_ir * w_ir], dim=1)  # [B, 2*C, H, W]

        return out
