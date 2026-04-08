# 文件: ultralytics/nn/modules/fusion/cmxfusion.py
# 来源: CMX (Cross-Modal Fusion Network, CVPR)
# 核心: CM-FRM 双向跨模态特征校准 + FFM 轻量化融合

import torch
import torch.nn as nn
import torch.nn.functional as F


class CM_FRM(nn.Module):
    """
    Cross-Modal Feature Rectification Module.
    使用空间和通道注意力对两路特征进行双向校准。
    """

    def __init__(self, in_channels):
        super().__init__()
        # 通道校准：4倍通道 -> 2倍通道
        self.mlp_c = nn.Sequential(
            nn.Linear(in_channels * 4, in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 2, in_channels * 2),
        )
        # 空间校准：2倍通道 -> 2 (每路一个空间权重图)
        self.conv_s = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1, bias=False),
        )
        # 可学习平衡权重
        self.lambda_c = nn.Parameter(torch.tensor(0.5))
        self.lambda_s = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_rgb, x_x):
        b, c, h, w = x_rgb.size()

        # --- 通道校准 ---
        avg_rgb = F.adaptive_avg_pool2d(x_rgb, 1).view(b, c)
        max_rgb = F.adaptive_max_pool2d(x_rgb, 1).view(b, c)
        avg_x = F.adaptive_avg_pool2d(x_x, 1).view(b, c)
        max_x = F.adaptive_max_pool2d(x_x, 1).view(b, c)

        y_c = torch.cat([avg_rgb, max_rgb, avg_x, max_x], dim=1)
        w_c = torch.sigmoid(self.mlp_c(y_c)).view(b, c * 2, 1, 1)
        w_c_rgb, w_c_x = torch.split(w_c, c, dim=1)

        rgb_rec_c = w_c_x * x_x
        x_rec_c = w_c_rgb * x_rgb

        # --- 空间校准 ---
        y_s = torch.cat([x_rgb, x_x], dim=1)
        w_s = torch.sigmoid(self.conv_s(y_s))
        w_s_rgb, w_s_x = torch.split(w_s, 1, dim=1)

        rgb_rec_s = w_s_x * x_x
        x_rec_s = w_s_rgb * x_rgb

        # --- 校准输出 ---
        out_rgb = x_rgb + self.lambda_c * rgb_rec_c + self.lambda_s * rgb_rec_s
        out_x = x_x + self.lambda_c * x_rec_c + self.lambda_s * x_rec_s

        return out_rgb, out_x


class CMXFusion(nn.Module):
    """
    CMX: Cross-Modal Fusion Network 多模态融合模块。
    输入两路同尺寸同通道特征 [x_rgb, x_mod]，
    经 CM-FRM 双向跨模态校准后，通过 FFM 轻量化融合输出单路特征。

    Args:
        in_channels (int): 单个模态的输入通道数（两路相同）。
    """

    def __init__(self, in_channels):
        super().__init__()
        self.cm_frm = CM_FRM(in_channels)

        # FFM：轻量化特征融合模块（Concat -> 1x1 Conv -> DWConv -> 1x1 Conv）
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        """x: List[Tensor]，[x_rgb, x_mod]，shape 相同。"""
        x_rgb, x_x = x[0], x[1]

        # 1. 跨模态双向校准
        rectified_rgb, rectified_x = self.cm_frm(x_rgb, x_x)

        # 2. 混合通道融合
        out = torch.cat([rectified_rgb, rectified_x], dim=1)
        return self.fusion(out)
