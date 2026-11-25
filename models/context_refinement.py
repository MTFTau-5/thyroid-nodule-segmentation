import torch
from torch import nn
from torch.nn import functional as F
from .encoder import ConvBlock
from .dsconv import DSConv


def roles(probability, radius=2):
    """Overlapping soft geometric roles, not anatomical tissue categories."""
    k = 2*radius + 1
    outer = F.max_pool2d(probability, k, 1, radius)
    inner = -F.max_pool2d(-probability, k, 1, radius)
    return torch.cat([inner, outer-inner, outer-probability], 1).clamp(0, 1)


class LocalCrossAttention(nn.Module):
    """Non-overlapping local windows, padded keys masked; no H*W global attention.

    A query reads the surrounding tissue in its shared window (not a sliding
    window centred separately on every pixel). Window size is at feature scale.
    """
    def __init__(self, channels, heads=4, window=9):
        super().__init__()
        if channels % heads or window < 1:
            raise ValueError('invalid local attention heads/window')
        self.heads, self.window = heads, window
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.role_embed = nn.Conv2d(3, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.relative_bias = nn.Parameter(torch.zeros(heads, (2*window-1)**2))
        yy, xx = torch.meshgrid(torch.arange(window), torch.arange(window), indexing='ij')
        positions = torch.stack([yy.flatten(), xx.flatten()], -1)
        relative = positions[:, None] - positions[None, :] + window-1
        self.register_buffer('relative_index', relative[..., 0]*(2*window-1)+relative[..., 1], persistent=False)

    def partition(self, x):
        b, c, h, w = x.shape
        s = self.window
        return x.reshape(b, c, h//s, s, w//s, s).permute(0, 2, 4, 3, 5, 1).reshape(-1, s*s, c)

    def forward(self, boundary, tissue, role):
        b, c, h, w = tissue.shape
        s = self.window
        ph, pw = (-h) % s, (-w) % s
        def split(x):
            return self.partition(F.pad(x, (0, pw, 0, ph))).reshape(-1, s*s, self.heads, c//self.heads).transpose(1, 2)
        q = split(self.q(boundary))
        k = split(self.k(tissue) + self.role_embed(role))
        v = split(self.v(tissue))
        valid = self.partition(F.pad(tissue.new_ones(b, 1, h, w), (0, pw, 0, ph))).squeeze(-1)
        bias = self.relative_bias[:, self.relative_index].unsqueeze(0).expand(q.shape[0], -1, -1, -1)
        bias = bias.masked_fill(~valid[:, None, None, :].bool(), float('-inf'))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        out = out.transpose(1, 2).reshape(-1, s*s, c)
        out = out.reshape(b, (h+ph)//s, (w+pw)//s, s, s, c).permute(0, 5, 1, 3, 2, 4)
        out = out.reshape(b, c, h+ph, w+pw)[:, :, :h, :w]
        return boundary + self.proj(out)


class BoundaryRefinement(nn.Module):
    def __init__(self, channels, mode='context', kernel=9, window=9, heads=4,
                 radius=2, detach_roles=True, steps=1):
        super().__init__()
        if mode not in ('none', 'cnn', 'snake', 'context'):
            raise ValueError('refinement must be none/cnn/snake/context')
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError('refinement steps must be a positive integer')
        self.mode, self.radius, self.detach_roles = mode, radius, detach_roles
        self.steps = steps if mode in ('snake', 'context') else 1
        if mode == 'none':
            return
        self.regular = ConvBlock(channels, channels)
        if mode == 'cnn':
            self.x = ConvBlock(channels, channels)
            self.y = ConvBlock(channels, channels)
        else:
            self.x = DSConv(channels, channels, kernel, 'x')
            self.y = DSConv(channels, channels, kernel, 'y')
        self.fuse = nn.Conv2d(3*channels, channels, 1)
        if mode == 'context':
            self.context = LocalCrossAttention(channels, heads, window)
        self.gate = nn.Conv2d(2*channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, coarse):
        if self.mode == 'none':
            return x, x
        for _ in range(self.steps):
            x, boundary = self.refine_once(x, coarse)
        return x, boundary

    def refine_once(self, x, coarse):
        boundary = self.fuse(torch.cat([self.regular(x), self.x(x), self.y(x)], 1))
        if self.mode == 'context':
            p = coarse.sigmoid()
            if self.detach_roles:
                p = p.detach()
            boundary = self.context(boundary, x, roles(p, self.radius))
        gate = self.gate(torch.cat([x, boundary], 1)).sigmoid()
        return x + gate*self.proj(boundary), boundary
