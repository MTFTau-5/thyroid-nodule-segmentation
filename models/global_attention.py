import torch
from torch import nn
from torch.nn import functional as F


class GlobalAttention(nn.Module):
    def __init__(self, cin, dim, heads=4, depth=1):
        super().__init__()
        if heads < 1 or dim % heads:
            raise ValueError('attention channels must be divisible by heads')
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise ValueError('attention depth must be a positive integer')
        self.proj = nn.Conv2d(cin, dim, 1)
        self.position = nn.Parameter(torch.zeros(1, dim, 16, 16))
        nn.init.trunc_normal_(self.position, std=.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim*4, dropout=0.,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        pos = F.interpolate(self.position, (h, w), mode='bilinear', align_corners=False)
        tokens = (x + pos).flatten(2).transpose(1, 2)
        return self.layers(tokens).transpose(1, 2).reshape(b, c, h, w)
