# MAC - HDNet（TGRS 2025）多尺度注意力校准块
# 迁移自 RT-20260901/RTDETR-main/ultralytics/nn/extra_modules/MAC.py（仅取 C2f 所需部分）
# 改动：固定权值卷积核由"模块级 .cuda() Parameter"改为 __init__ 内 CPU 张量赋值（设备安全，行为不变）

import numpy as np
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


def _gen_center_kernels():
    """生成 HDNet 固定权值中心-环绕卷积核（k=3,5,7,9,11 各若干组）。"""
    kernels_all = []
    for i in range(1, 6):
        kernels = []
        for j in range(i):
            k_size = (2 * i) + 1
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32)
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2
            red_size = (j + 1) * 2 - 1
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size)
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[:, lt_y + red_size:] = blue_val
            kernels.append(kernel)
        kernels_all.append(kernels)
    return [k for ks in kernels_all for k in ks]


def _gen_avg_kernel():
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    return [kernel / 9.0]


def _gen_laplace_kernel():
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    kernel = kernel / 8.0 * -1
    kernel[1, 1] = 0
    return [kernel]


_FIXED_KERNELS = _gen_center_kernels()
_AVG_KERNELS = _gen_avg_kernel()
_LAPLACE_KERNELS = _gen_laplace_kernel()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, max(1, in_planes // 16), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(max(1, in_planes // 16), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MAC(nn.Module):
    def __init__(self, inplanes, outplanes, one=3, two=3, three=3, scales=4):
        super(MAC, self).__init__()
        if outplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        self.scales = scales
        self.relu = nn.ReLU(inplace=True)
        self.spx = outplanes // scales
        self.inconv = Conv(inplanes, outplanes, act=False)

        def fixed(k):
            return torch.from_numpy(k).unsqueeze(0).unsqueeze(0)

        self.conv1 = Conv(self.spx, self.spx, k=one, g=self.spx, act=False)
        self.conv1.conv.weight.data = fixed(_FIXED_KERNELS[one // 2 - 1]).repeat(self.spx, 1, 1, 1)

        self.conv2 = Conv(self.spx, self.spx, k=one, g=self.spx, act=False)
        self.conv2.conv.weight.data = fixed(_FIXED_KERNELS[two // 2 - 1]).repeat(self.spx, 1, 1, 1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, three, 1, 1, groups=self.spx),
        )
        self.conv3[0].weight.data = fixed(_AVG_KERNELS[0]).repeat(self.spx, 1, 1, 1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, three, 1, 2, groups=self.spx, dilation=2),
        )
        self.conv4[0].weight.data = fixed(_LAPLACE_KERNELS[0]).repeat(self.spx, 1, 1, 1)

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(self.spx)
        )
        self.outconv = Conv(outplanes, outplanes, k=3, act=nn.ReLU)
        self.ca = ChannelAttention(outplanes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.inconv(x)
        input = x
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(xs[0])
        ys.append(self.relu(self.conv1(xs[1])))
        ys.append(self.relu(self.conv2(xs[2] + ys[1])))
        temp = xs[3] + ys[2]
        temp1 = self.conv5(self.conv3(temp) + self.conv4(temp))
        ys.append(self.relu(temp1))
        y = torch.cat(ys, 1)

        y = self.outconv(y)

        output = self.relu(y + input)
        return output
