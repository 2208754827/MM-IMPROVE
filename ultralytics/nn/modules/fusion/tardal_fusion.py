# 文件: ultralytics/nn/modules/fusion/tardal_fusion.py
# 来源: TarDAL (CVPR 2022) — Target-aware Dual Adversarial Learning for Infrared-Visible Fusion
# 核心: 目标感知空间权重融合（Spatial Weight Map，红外显著性优先）
# 二次创新:
#   1. Laplacian 梯度分支 —— 对可见光做拉普拉斯算子提取背景纹理梯度，增强细节互补性
#   2. 双权重解耦 (Dual-Weight) —— 红外/可见光各自拥有独立空间权重图，互不干扰
#   3. 空洞卷积扩大感受野 —— spatial_gate 引入多尺度空洞卷积，更好捕捉小目标整体结构

import torch
import torch.nn as nn
import torch.nn.functional as F


class TarDALFusion(nn.Module):
    """
    TarDAL: Target-aware Dual Adversarial Learning 推理融合模块（二次创新增强版）。

    三阶段设计：
      Stage 1 (Laplacian 梯度增强): 对可见光分支做拉普拉斯边缘提取，
            强化背景纹理梯度信息，防止细节被红外热目标掩盖。
      Stage 2 (双权重解耦 Dual-Weight): 利用多尺度空洞卷积 spatial_gate
            分别为红外和可见光（含梯度增强）生成独立空间权重图，
            避免单一权重导致两路特征相互干扰。
      Stage 3 (加权融合 + DWConv平滑): w_ir × x_ir + w_vis × x_vis_enh，
            DWConv 增强特征一致性，BN 稳定 AMP fp16 训练。

    输入: List[Tensor] -> [x_ir, x_vis]，shape 相同 [B, C, H, W]
    输出: Tensor，shape [B, C, H, W]（通道数不变，无膨胀）

    Args:
        c1 (int): 两路输入通道数（应相同）。
    """

    def __init__(self, c1):
        super().__init__()
        self.c = c1
        mid = max(1, c1 // 4)

        # ---- Stage 1: Laplacian 梯度提取（可见光分支）----
        # 固定 Laplacian 核，检测背景边缘/纹理梯度（无学习参数，零额外开销）
        laplacian_kernel = torch.tensor(
            [[0,  1, 0],
             [1, -4, 1],
             [0,  1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        # 扩展到 c1 通道（depthwise），注册为 buffer 不参与梯度计算
        self.register_buffer(
            'lap_kernel',
            laplacian_kernel.repeat(c1, 1, 1, 1)  # [c1, 1, 3, 3]
        )

        # 梯度特征融合投影：将梯度残差投影回 c1
        self.grad_proj = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # ---- Stage 2: 双权重解耦 Dual-Weight Spatial Gate ----
        # 使用多尺度空洞卷积扩大感受野，增强小目标结构感知
        # 红外权重图（目标显著性）
        self.gate_ir = nn.Sequential(
            nn.Conv2d(c1 * 2, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            # 空洞卷积：rate=2，感受野 5x5
            nn.Conv2d(mid, mid, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            # 空洞卷积：rate=4，感受野 9x9
            nn.Conv2d(mid, mid, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        # 可见光权重图（纹理细节显著性）
        self.gate_vis = nn.Sequential(
            nn.Conv2d(c1 * 2, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            # 空洞卷积：rate=2
            nn.Conv2d(mid, mid, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            # 空洞卷积：rate=4
            nn.Conv2d(mid, mid, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # ---- Stage 3: DWConv 平滑 + BN 稳定 ----
        self.smooth = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.bn_out = nn.BatchNorm2d(c1)

    def _laplacian_grad(self, x):
        """
        对输入特征图做 depthwise Laplacian 卷积，提取边缘梯度图。
        padding=1 保持空间分辨率不变。kernel 自动跟随输入 dtype（fp16/fp32 均兼容）。
        """
        return F.conv2d(
            x,
            self.lap_kernel.to(dtype=x.dtype, device=x.device),
            padding=1,
            groups=self.c,
        )

    def forward(self, x):
        """
        Args:
            x: List[Tensor]，[x_ir, x_vis]，shape 需相同 [B, C, H, W]。
        Returns:
            Tensor: 融合后特征，shape [B, C, H, W]。
        """
        x_ir, x_vis = x[0], x[1]

        # ---- Stage 1: Laplacian 梯度增强可见光分支 ----
        # 提取可见光背景纹理梯度（绝对值取正，突出边缘幅度）
        vis_grad = torch.abs(self._laplacian_grad(x_vis))
        # 投影融合梯度信息（残差形式，保留原始特征）
        x_vis_enh = x_vis + self.grad_proj(vis_grad)

        # ---- Stage 2: 双权重解耦 ----
        # 拼接红外与增强可见光，分别生成独立空间权重图
        feat_cat = torch.cat([x_ir, x_vis_enh], dim=1)  # [B, 2C, H, W]
        w_ir = self.gate_ir(feat_cat)    # [B, 1, H, W] 红外显著性权重
        w_vis = self.gate_vis(feat_cat)  # [B, 1, H, W] 可见光细节权重

        # ---- Stage 3: 加权融合 ----
        # 双权重独立加权（不强制 w_ir + w_vis = 1，允许自适应双路增强）
        out = w_ir * x_ir + w_vis * x_vis_enh

        # DWConv 增强空间一致性 + 残差连接 + BN 稳定
        out = self.smooth(out) + out
        return self.bn_out(out)
