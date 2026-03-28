import torch
import torch.nn as nn
import einops


class SpatialAttentionCGA(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        return self.sa(torch.cat([x_avg, x_max], dim=1))


class ChannelAttentionCGA(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        mid = max(dim // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=True),
        )

    def forward(self, x):
        return self.ca(self.gap(x))


class PixelAttentionCGA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode="reflect", groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        x = x.unsqueeze(2)
        pattn1 = pattn1.unsqueeze(2)
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = einops.rearrange(x2, "b c t h w -> b (c t) h w")
        return self.sigmoid(self.pa(x2))


class CGAFusion(nn.Module):
    """Channel/Spatial/Pixel attention guided two-input fusion."""

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.sa = SpatialAttentionCGA()
        self.ca = ChannelAttentionCGA(dim, reduction=reduction)
        self.pa = PixelAttentionCGA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data
        initial = x + y
        pattn1 = self.sa(initial) + self.ca(initial)
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        out = initial + pattn2 * x + (1.0 - pattn2) * y
        return self.conv(out)
