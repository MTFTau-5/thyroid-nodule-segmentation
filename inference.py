"""Probability-space flip/scale ensembling for evaluation and prediction."""
import math

import torch
from torch.nn import functional as F


def validate_scales(scales):
    scales = tuple(float(scale) for scale in scales)
    if not scales or any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError('TTA scales must be finite and positive')
    return scales


@torch.inference_mode()
def predict_probability(model, image, scales=(1.,), horizontal_flip=False, output=None):
    scales = validate_scales(scales)
    shape = image.shape[-2:]
    probability_sum, count = None, 0
    for scale in scales:
        scaled_shape = shape if scale == 1. else tuple(max(32, round(side*scale/32)*32) for side in shape)
        scaled = image if scaled_shape == shape else F.interpolate(
            image, size=scaled_shape, mode='bilinear', align_corners=False)
        for flipped in ((False, True) if horizontal_flip else (False,)):
            if not flipped and scaled_shape == shape and output is not None:
                prediction = output
            else:
                prediction = model(scaled.flip(-1) if flipped else scaled)
            probability = prediction['segmentation'].float().sigmoid()
            if flipped:
                probability = probability.flip(-1)
            if probability.shape[-2:] != shape:
                probability = F.interpolate(probability, size=shape, mode='bilinear', align_corners=False)
            probability_sum = probability if probability_sum is None else probability_sum + probability
            count += 1
    return probability_sum / count
