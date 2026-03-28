import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class PSPModule(nn.Module):
    def __init__(self, grids=(1, 2, 3, 6), channels=256):
        super().__init__()
        self.grids = grids
        self.channels = channels

    def forward(self, feats):
        b, _, h, w = feats.size()
        ar = w / h
        pooled = []
        for g in self.grids:
            gh = g
            gw = max(1, round(ar * g))
            pooled.append(F.adaptive_avg_pool2d(feats, (gh, gw)).view(b, self.channels, -1))
        return torch.cat(pooled, dim=2)


class LocalAttenModule(nn.Module):
    def __init__(self, in_channels=256, inter_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(in_channels, inter_channels, 1),
            nn.Conv2d(inter_channels, in_channels, kernel_size=3, padding=1, bias=False),
        )
        self.tanh_spatial = nn.Tanh()
        self.conv[1].weight.data.zero_()

    def forward(self, x):
        x_mask = self.tanh_spatial(self.conv(x))
        return x * x_mask + x


class CFC_CRB(nn.Module):
    def __init__(self, in_channels=512, grids=(6, 3, 2, 1)):
        super().__init__()
        inter_channels = max(in_channels // 2, 1)
        self.inter_channels = inter_channels
        self.reduce_channel = Conv(in_channels, inter_channels, 3)
        self.query_conv = nn.Conv2d(inter_channels, 32, kernel_size=1)
        self.key_conv = nn.Conv1d(inter_channels, 32, kernel_size=1)
        self.value_conv = nn.Conv1d(inter_channels, inter_channels, kernel_size=1)
        self.value_psp = PSPModule(grids, inter_channels)
        self.key_psp = PSPModule(grids, inter_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.local_attention = LocalAttenModule(inter_channels, max(inter_channels // 8, 1))

    def forward(self, x):
        x = self.reduce_channel(x)
        b, _, h, w = x.size()
        query = self.query_conv(x).view(b, 32, -1).permute(0, 2, 1)
        key = self.key_conv(self.key_psp(x))
        sim_map = self.softmax(torch.matmul(query, key))
        value = self.value_conv(self.value_psp(x))
        context = torch.bmm(value, sim_map.permute(0, 2, 1)).view(b, self.inter_channels, h, w)
        context = self.local_attention(context)
        return x + context


class SFC_G2(nn.Module):
    def __init__(self, inc):
        super().__init__()
        hidc = inc[0]
        self.groups = 2
        self.conv_8 = Conv(inc[0], hidc, 3)
        self.conv_32 = Conv(inc[1], hidc, 3)
        self.conv_offset = nn.Sequential(
            Conv(hidc * 2, 64),
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False),
        )
        self.conv_offset[1].weight.data.zero_()

    def forward(self, x):
        cp, sp = x
        n, _, out_h, out_w = cp.size()
        sp = self.conv_32(sp)
        sp = F.interpolate(sp, cp.size()[2:], mode="bilinear", align_corners=True)
        cp = self.conv_8(cp)
        conv_results = self.conv_offset(torch.cat([cp, sp], 1))

        sp = sp.reshape(n * self.groups, -1, out_h, out_w)
        cp = cp.reshape(n * self.groups, -1, out_h, out_w)

        offset_l = conv_results[:, 0 : self.groups * 2, :, :].reshape(n * self.groups, -1, out_h, out_w)
        offset_h = conv_results[:, self.groups * 2 : self.groups * 4, :, :].reshape(n * self.groups, -1, out_h, out_w)

        norm = torch.tensor([[[[out_w, out_h]]]], device=sp.device, dtype=sp.dtype)
        w_grid = torch.linspace(-1.0, 1.0, out_h, device=sp.device, dtype=sp.dtype).view(-1, 1).repeat(1, out_w)
        h_grid = torch.linspace(-1.0, 1.0, out_w, device=sp.device, dtype=sp.dtype).repeat(out_h, 1)
        grid = torch.cat((h_grid.unsqueeze(2), w_grid.unsqueeze(2)), 2)
        grid = grid.repeat(n * self.groups, 1, 1, 1)

        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm

        cp = F.grid_sample(cp, grid_l, align_corners=True)
        sp = F.grid_sample(sp, grid_h, align_corners=True)

        cp = cp.reshape(n, -1, out_h, out_w)
        sp = sp.reshape(n, -1, out_h, out_w)

        att = 1 + torch.tanh(conv_results[:, self.groups * 4 :, :, :])
        sp = sp * att[:, 0:1, :, :] + cp * att[:, 1:2, :, :]
        return sp
