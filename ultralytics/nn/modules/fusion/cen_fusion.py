# 文件: ultralytics/nn/modules/fusion/cen_fusion.py
# 来源: CEN (Channel Exchange Networks), TPAMI 2022
# 参考: https://github.com/yikaiw/CEN

import torch
import torch.nn as nn


class CENFusion(nn.Module):
    """
    受 TPAMI 2022 CEN (Channel Exchange Networks) 启发的零参数融合模块。
    将两路同尺寸特征图的部分通道进行物理交换后相加融合。
    零参数量，零计算量增量，完美平替 Concat + 1x1 Conv 降维。

    Args:
        c (int): 输入特征通道数（两路相同）。
        p (float): 交换的通道比例，默认 0.5（交换一半通道）。
    """

    def __init__(self, c, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        x: 包含两个模态特征的列表 [x_rgb, x_mod]，shape 相同。
        返回: 通道交换后相加的融合特征，通道数不变。
        """
        x1, x2 = x[0], x[1]
        assert x1.shape == x2.shape, "CENFusion requires inputs to have the same shape"

        c = x1.size(1)
        exchange_c = int(c * self.p)

        # 物理通道交换 (Channel Exchange)
        out1 = torch.cat((x2[:, :exchange_c], x1[:, exchange_c:]), dim=1)
        out2 = torch.cat((x1[:, :exchange_c], x2[:, exchange_c:]), dim=1)

        # 相加融合，通道数保持不变（256）
        return out1 + out2
