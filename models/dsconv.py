"""Differentiable 2-D snake sampling, inspired by DSCNet (ICCV 2023).

Independent reimplementation: centre offset is zero; both sides accumulate
increments outwards INCLUDING endpoints. No detach/in-place copies of offsets.
This implements the continuous sampling idea, not DSCNet's complete network
or topology loss. Border sampling uses grid_sample's border replication.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .encoder import groups


class DSConv(nn.Module):
    def __init__(self, cin, cout, kernel_size=9, axis='x', extend_scope=1., deform=True):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 != 1 or axis not in ('x', 'y'):
            raise ValueError('DSConv requires an odd kernel >=3 and axis x/y')
        self.kernel_size, self.axis = kernel_size, axis
        self.extend_scope, self.deform = extend_scope, deform
        self.offset = nn.Conv2d(cin, kernel_size, 3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.aggregate = nn.Conv2d(cin, cout, (1, kernel_size), stride=(1, kernel_size), bias=False)
        self.norm = nn.GroupNorm(groups(cout), cout)

    def cumulative_offsets(self, x):
        raw = self.offset(x).tanh() * self.extend_scope
        c = self.kernel_size // 2
        if not self.deform:
            return raw * 0
        left = raw[:, :c].flip(1).cumsum(1).flip(1)
        right = raw[:, c+1:].cumsum(1)
        return torch.cat([left, torch.zeros_like(raw[:, c:c+1]), right], 1)

    def sampling_grid(self, x):
        b, _, h, w = x.shape
        # Grid calculations in fp32 also under mixed precision.
        delta = self.cumulative_offsets(x).float().permute(0, 2, 3, 1)
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=torch.float32),
                                torch.arange(w, device=x.device, dtype=torch.float32), indexing='ij')
        spread = torch.arange(self.kernel_size, device=x.device).float() - self.kernel_size // 2
        gx = xx[None, :, :, None].expand(b, h, w, self.kernel_size)
        gy = yy[None, :, :, None].expand_as(gx)
        if self.axis == 'x':
            gx, gy = gx + spread, gy + delta
        else:
            gx, gy = gx + delta, gy + spread
        # align_corners=False handles singleton dimensions as well.
        return torch.stack([2 * (gx + .5) / w - 1, 2 * (gy + .5) / h - 1], -1).reshape(b, h, w*self.kernel_size, 2)

    def forward(self, x):
        grid = self.sampling_grid(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            sampled = F.grid_sample(x.float(), grid, mode='bilinear',
                                    padding_mode='border', align_corners=False)
        return F.silu(self.norm(self.aggregate(sampled.to(x.dtype))))
