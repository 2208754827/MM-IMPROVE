# 文件: ultralytics/nn/modules/fusion/metafusion.py
# 来源: MetaFusion (wdzhao123/MetaFusion)
# 核心思想: 使用检测分支语义特征 (Fej) 生成元特征，引导融合分支 (Fuj)
# 残差融合: Fout = Fuj + Ftj，语义信息无损注入

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..conv import Conv


class MetaFeatureFusion(nn.Module):
    """
    MetaFusion: Meta-Feature Embedding 语义增强融合模块。
    输入两路同尺度特征图 [f_uj (融合分支), f_ej (检测/语义分支)]，
    利用检测分支语义先验生成元特征，残差注入融合分支，输出通道与融合分支一致。

    Args:
        c1 (int): 融合分支输入通道数 (Fuj)。
        c2 (int): 语义分支输入通道数 (Fej)。
    """

    def __init__(self, c1, c2):
        super().__init__()

        # 1. 语义特征对齐：将 Fej 通道数映射到 c1
        self.od_conv = Conv(c2, c1, 3)

        # 2. 元特征生成器 MFG：拼接后通过 3 层卷积提取元特征
        self.mfg_conv = nn.Sequential(
            Conv(c1 * 2, c1, 3),
            Conv(c1, c1, 3),
            Conv(c1, c1, 3),
        )

        # 3. 特征变换器 FT：3 层卷积生成特征桥梁 Ftj
        self.ft = nn.Sequential(
            Conv(c1, c1, 3),
            Conv(c1, c1, 3),
            Conv(c1, c1, 3),
        )

    def forward(self, x):
        """
        x: List[Tensor]，[f_uj (融合特征), f_ej (语义特征)]
        返回: f_uj + Ftj，通道数不变 (= c1)
        """
        f_uj, f_ej = x[0], x[1]

        # 尺寸对齐（若语义分支分辨率不同则上采样）
        if f_uj.shape[2:] != f_ej.shape[2:]:
            f_ej = F.interpolate(f_ej, size=f_uj.shape[2:], mode='bilinear', align_corners=True)

        # 语义特征通道对齐
        f_ej_proc = self.od_conv(f_ej)

        # 元特征生成
        meta_feat = self.mfg_conv(torch.cat([f_uj, f_ej_proc], dim=1))

        # 特征桥梁变换
        f_tj = self.ft(meta_feat)

        # 残差融合：Fout = Fuj + Ftj
        return f_uj + f_tj
