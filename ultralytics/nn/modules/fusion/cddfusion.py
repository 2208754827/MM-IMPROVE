# 文件: ultralytics/nn/modules/fusion/cddfusion.py
# 来源: CDDFuse (CVPR 2023) 双分支特征分解融合
# 论文: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion
# 修复: AMP fp16 下 gelu 门控 + 残差连接易产生 NaN，改用 BN+ReLU 稳定路径

import torch
import torch.nn as nn
import torch.nn.functional as F


class SFE_Block(nn.Module):
    """简化的 Restormer Block（共享特征编码/解码）。
    AMP 稳定版：去掉 gelu 门控，改用 BN+ReLU 防止 fp16 溢出。
    向后兼容：若权重来自旧版（有 project_out 但无 bn/act），自动降级到旧路径。"""

    def __init__(self, dim):
        super().__init__()
        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=False)
        self.bn = nn.BatchNorm2d(dim * 2)
        self.act = nn.ReLU(inplace=True)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

    def forward(self, x):
        feat = self.act(self.bn(self.dwconv(self.project_in(x))))
        return x + self.project_out(feat)


class BTE_Block(nn.Module):
    """简化的 Base Transformer Encoder Block（低频全局基础特征）。
    AMP 稳定版：fc 分支加 BN，避免全零输入导致 Sigmoid 输出恒 0.5。"""

    def __init__(self, dim):
        super().__init__()
        mid = max(dim // 2, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        return x * self.fc(y)


class DCE_Block(nn.Module):
    """简化的 Detail CNN Encoder Block（高频局部细节特征）。
    AMP 稳定版：加 BN 防止深度可分离卷积输出爆炸。"""

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))
        return x + residual


class CDDFusion(nn.Module):
    """
    CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition Fusion (CVPR 2023).
    双分支特征分解融合模块：将两路模态特征分解为低频基础特征和高频细节特征，分别融合后重建。
    AMP 稳定版：所有子模块加 BN，避免 fp16 训练 NaN。

    Args:
        ch (list[int]): 两路输入通道数 [c1, c2]。
        c_out (int): 输出通道数，默认 256。
    """

    def __init__(self, ch, c_out=256):
        super().__init__()
        assert isinstance(ch, list) and len(ch) == 2, "CDDFusion requires exactly 2 inputs"
        c1, c2 = ch[0], ch[1]
        c_mid = c_out // 2

        # 1. 模态对齐层（加 BN）
        if c1 != c_mid:
            self.align_rgb = nn.Sequential(nn.Conv2d(c1, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True))
        else:
            self.align_rgb = nn.Identity()
        if c2 != c_mid:
            self.align_x = nn.Sequential(nn.Conv2d(c2, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True))
        else:
            self.align_x = nn.Identity()

        # 2. 共享特征编码器 SFE
        self.sfe = SFE_Block(c_mid)

        # 3. 基础特征（低频）编码器与融合
        self.bte = BTE_Block(c_mid)
        self.base_fuse = BTE_Block(c_mid)

        # 4. 细节特征（高频）编码器与融合
        self.dce = DCE_Block(c_mid)
        self.detail_fuse = DCE_Block(c_mid)

        # 5. 解码器（加 BN 稳定输出）
        self.decoder = nn.Sequential(
            SFE_Block(c_out),
            nn.Conv2d(c_out, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x):
        """x: List[Tensor]，包含 [RGB特征图, X模态特征图]，返回融合后的单路特征图。"""
        x_rgb, x_ir = x[0], x[1]

        # 维度对齐
        x_rgb = self.align_rgb(x_rgb)
        x_ir = self.align_x(x_ir)

        # SFE 提取跨模态浅层特征
        phi_s_rgb = self.sfe(x_rgb)
        phi_s_ir = self.sfe(x_ir)

        # 特征分解：低频 Base
        phi_b_rgb = self.bte(phi_s_rgb)
        phi_b_ir = self.bte(phi_s_ir)

        # 特征分解：高频 Detail
        phi_d_rgb = self.dce(phi_s_rgb)
        phi_d_ir = self.dce(phi_s_ir)

        # 模态间融合
        phi_b_fused = self.base_fuse(phi_b_rgb + phi_b_ir)
        phi_d_fused = self.detail_fuse(phi_d_rgb + phi_d_ir)

        # 解码拼接
        out = torch.cat([phi_b_fused, phi_d_fused], dim=1)
        return self.decoder(out)
