"""CNN encoder/decoder; 512 input -> 128, 64, 32, 16 feature grids."""
import torch
from torch import nn
from torch.nn import functional as F


def groups(channels):
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.GroupNorm(groups(cout), cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups(cout), cout), nn.SiLU())


class Encoder(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, base // 2, 3, 2, 1), nn.SiLU())
        self.blocks = nn.ModuleList([ConvBlock(base // 2, base, 2)] + [
            ConvBlock(base * 2**i, base * 2**(i+1), 2) for i in range(3)])

    def forward(self, x):
        x = self.stem(x)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class Decoder(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(base * 4 + base * 4, base * 4),
            ConvBlock(base * 4 + base * 2, base * 2),
            ConvBlock(base * 2 + base, base)])

    def forward(self, x, skips, mid_refine=None, mid_head=None, return_coarse=False):
        intermediate_coarse = None
        for index, (block, skip) in enumerate(zip(self.blocks, reversed(skips))):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
            if index == 1 and mid_refine is not None:
                intermediate_coarse = mid_head(x)
                x, _ = mid_refine(x, intermediate_coarse)
        return (x, intermediate_coarse) if return_coarse else x
