import torch
from torch import nn
from torch.nn import functional as F


def boundary_target(mask, radius=2):
    k = 2*radius+1
    dilated = F.max_pool2d(mask, k, 1, radius)
    eroded = -F.max_pool2d(-mask, k, 1, radius)
    return (dilated-eroded).clamp(0,1)


def dice_bce(logits, target, valid, balance=False):
    # Explicit fp32 reductions prevent fp16 overflow for 512x512 masks.
    logits, target, valid = logits.float(), target.float(), valid.float()
    weight = None
    if balance:
        positives = (target*valid).sum().detach()
        negatives = ((1-target)*valid).sum().detach()
        weight = (negatives/positives.clamp_min(1)).clamp(1,20)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none', pos_weight=weight)
    bce = (bce*valid).sum()/valid.sum().clamp_min(1)
    p, y = logits.sigmoid()*valid, target*valid
    dims = (1,2,3)
    dice = (2*(p*y).sum(dims)+1)/(p.sum(dims)+y.sum(dims)+1)
    return bce + 1-dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, coarse_weight=.4, boundary_weight=.2, boundary_radius=2):
        super().__init__()
        self.coarse_weight, self.boundary_weight = coarse_weight, boundary_weight
        self.boundary_radius = boundary_radius

    def forward(self, output, target, valid):
        seg = dice_bce(output['segmentation'], target, valid)
        coarse = dice_bce(output['coarse'], target, valid)
        if 'intermediate_coarse' in output:
            coarse = (coarse + dice_bce(output['intermediate_coarse'], target, valid)) / 2
        edge = boundary_target(target, self.boundary_radius)
        boundary = dice_bce(output['boundary'], edge, valid, balance=True)
        total = seg+self.coarse_weight*coarse+self.boundary_weight*boundary
        return total, {'segmentation':seg.detach(), 'coarse':coarse.detach(), 'boundary':boundary.detach()}
