import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class GlobalExtraction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        return self.proj(torch.cat((avg, mx), dim=1))


class ContextExtraction(nn.Module):
    def __init__(self, dim: int, reduction: int = 1) -> None:
        super().__init__()
        red = max(int(reduction), 1)
        out_dim = max(dim // red, 1)
        self.dconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dconv(x))


class MultiscaleFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.local = ContextExtraction(dim)
        self.global_ = GlobalExtraction()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = self.local(x)
        g = self.global_(g)
        return self.bn(x + g)


class MultiScaleGatedAttn(nn.Module):
    def __init__(self, dims):
        super().__init__()
        dim = min(dims)
        if dims[0] != dims[1]:
            self.conv1 = Conv(dims[0], dim)
            self.conv2 = Conv(dims[1], dim)
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv2d(dim, 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1))

    def forward(self, inputs):
        x, g = inputs
        if x.size(1) != g.size(1):
            x = self.conv1(x)
            g = self.conv2(g)

        x_ = x
        g_ = g
        multi = self.multi(x, g)
        attn = F.softmax(self.selection(multi), dim=1)
        a, b = attn.split(1, dim=1)

        x_att = a.expand_as(x_) * x_ + x_
        g_att = b.expand_as(g_) * g_ + g_

        x_att_2 = torch.sigmoid(g_att) * x_att
        g_att_2 = torch.sigmoid(x_att) * g_att
        interaction = x_att_2 * g_att_2

        projected = torch.sigmoid(self.bn(self.proj(interaction)))
        weighted = projected * x_
        y = self.conv_block(weighted)
        y = self.bn2(y)
        return y
