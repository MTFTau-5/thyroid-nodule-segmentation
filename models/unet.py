from torch import nn
from torch.nn import functional as F
from .encoder import Encoder, Decoder
from .global_attention import GlobalAttention
from .context_refinement import BoundaryRefinement
from .heads import SegmentationHeads


class ThyroidSegmentation(nn.Module):
    def __init__(self, base=64, global_attention=True, heads=4, depth=1,
                 refinement='context', snake_kernel=9, window=9, radius=2, detach_roles=True,
                 multiscale_context=False, refinement_steps=1):
        super().__init__()
        if base < 8 or base % 8:
            raise ValueError('base channels must be a positive multiple of 8')
        if multiscale_context and refinement != 'context':
            raise ValueError('multiscale_context requires context refinement')
        self.encoder = Encoder(base)
        self.global_context = (GlobalAttention(8*base, 4*base, heads, depth)
                               if global_attention else nn.Conv2d(8*base, 4*base, 1))
        self.decoder = Decoder(base)
        self.refine = BoundaryRefinement(base, refinement, snake_kernel, window, heads,
                                         radius, detach_roles, refinement_steps)
        self.mid_head = nn.Conv2d(2*base, 1, 1) if multiscale_context else None
        self.mid_refine = (BoundaryRefinement(2*base, refinement, snake_kernel, window, heads,
                                             radius, detach_roles, refinement_steps)
                           if multiscale_context else None)
        self.heads = SegmentationHeads(base)

    def forward(self, image):
        shape = image.shape[-2:]
        if min(shape) < 32:
            raise ValueError('input sides must be >=32')
        features = self.encoder(image)
        u, intermediate_coarse = self.decoder(self.global_context(features[-1]), features[:-1],
                                               self.mid_refine, self.mid_head, return_coarse=True)
        coarse = self.heads.coarse(u)
        refined, boundary = self.refine(u, coarse)
        resize = lambda t: F.interpolate(t, shape, mode='bilinear', align_corners=False)
        output = {'segmentation': resize(self.heads.segmentation(refined)),
                  'coarse': resize(coarse), 'boundary': resize(self.heads.boundary(boundary))}
        if intermediate_coarse is not None:
            output['intermediate_coarse'] = resize(intermediate_coarse)
        return output
