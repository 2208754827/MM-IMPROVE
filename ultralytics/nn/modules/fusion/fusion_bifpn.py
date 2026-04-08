# 文件: ultralytics/nn/modules/fusion/fusion_bifpn.py
# 来源: RTDETR-main block.py - Fusion(bifpn模式)

import torch
import torch.nn as nn


class FusionBiFPN(nn.Module):
    """
    BiFPN-style 可学习标量加权融合。
    对两路同尺寸特征图用正规化可学习权重加权求和。
    参数极少，训练稳定，适合轻量化多模态 P3 融合。

    Args:
        inc (list[int]): 两路输入通道数，如 [256, 256]。
    """

    def __init__(self, inc):
        super().__init__()
        assert len(inc) == 2, "FusionBiFPN expects exactly 2 input channels"
        # 若通道不一致则先对齐
        self.align = None
        if inc[0] != inc[1]:
            self.align = nn.Conv2d(inc[0], inc[1], kernel_size=1, bias=False)

        # 两个可学习融合权重（初始化为1）
        self.fusion_weight = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.relu = nn.ReLU()
        self.epsilon = 1e-4

    def forward(self, x):
        x1, x2 = x
        if self.align is not None:
            x1 = self.align(x1)
        # 正规化权重
        w = self.relu(self.fusion_weight.clone())
        w = w / (torch.sum(w) + self.epsilon)
        return w[0] * x1 + w[1] * x2
