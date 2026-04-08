# 文件: ultralytics/nn/modules/fusion/superyolofusion.py
# 来源: SuperYOLO (TGRS 2023) - Multimodal Pixel-level Fusion (MF) 模块
# 核心: 对称SE通道注意力 + 跨模态空间掩码双向交互 + 残差连接 + 全局SE降维

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    提取模态内部的通道注意力权重。
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        reduction_c = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction_c, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_c, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SuperYOLOFusion(nn.Module):
    """
    基于 SuperYOLO (TGRS 2023) 提出的 Multimodal Fusion (MF) 模块。
    执行对称的双向像素级特征融合，无缝平替 Concat + 1x1 Conv。

    流程：
      1. 对称SE通道注意力提取各模态内部信息 (f_rgb, f_ir)
      2. 生成跨模态空间掩码 (m_rgb, m_ir)
      3. 跨模态逐元素矩阵乘法：f_rgb * m_ir，f_ir * m_rgb
      4. 残差残差相加后经 1x1 Conv 混合
      5. Concat -> 全局SE校准 -> 1x1 Conv 降维输出

    Args:
        in_channels (int): 单个模态的输入通道数（两路需相同）。
    """

    def __init__(self, in_channels):
        super().__init__()

        # 1. 独立提取各模态通道维度内部信息
        self.se_rgb = SEBlock(in_channels)
        self.se_ir = SEBlock(in_channels)

        # 2. 生成用于跨模态的 Spatial 权重 (论文公式 2)
        self.conv_m_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv_m_ir = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # 3. 融合后的非线性变换 (论文公式 4)
        self.conv_f_rgb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.conv_f_ir = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        # 4. 拼接后总体通道校准与降维 (论文公式 5)
        self.se_out = SEBlock(in_channels * 2)
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        """x: List[Tensor]，[x_rgb, x_ir]，shape 需相同。"""
        x_rgb, x_ir = x[0], x[1]

        # 步骤 1: 提取各模态通道内部信息
        f_rgb = self.se_rgb(x_rgb)
        f_ir = self.se_ir(x_ir)

        # 步骤 2: 生成跨模态空间掩码
        m_rgb = torch.sigmoid(self.conv_m_rgb(f_rgb))
        m_ir = torch.sigmoid(self.conv_m_ir(f_ir))

        # 步骤 3: 跨模态逐元素矩阵乘法（Cross Element-wise）
        f_in_rgb = f_rgb * m_ir
        f_in_ir = f_ir * m_rgb

        # 步骤 4: 残差相加后执行 1x1 Conv 混合
        f_ful_rgb = self.conv_f_rgb(f_in_rgb + x_rgb)
        f_ful_ir = self.conv_f_ir(f_in_ir + x_ir)

        # 步骤 5: 最终融合（Concat -> SE校准 -> Conv降维）
        out = torch.cat([f_ful_rgb, f_ful_ir], dim=1)
        out = self.se_out(out)
        out = self.conv_out(out)

        return out
