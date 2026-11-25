"""Training-only ultrasound perturbations applied before letterboxing."""
import random

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates


class UltrasoundAugmentation:
    def __init__(self, elastic_prob=.25, elastic_alpha=.02, elastic_sigma=.04,
                 speckle_prob=.35, speckle_std=.15, gain_prob=.35, gain_range=(.8, 1.2),
                 attenuation_max=.35, bright_spot_prob=.2, bright_spot_strength=.25):
        probabilities = (elastic_prob, speckle_prob, gain_prob, bright_spot_prob)
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError('augmentation probabilities must be in [0,1]')
        strengths = (elastic_alpha, speckle_std, attenuation_max, bright_spot_strength)
        if any(not np.isfinite(value) or value < 0 for value in strengths):
            raise ValueError('augmentation strengths must be finite and nonnegative')
        if not np.isfinite(elastic_sigma) or elastic_sigma <= 0:
            raise ValueError('elastic_sigma must be finite and positive')
        if (len(gain_range) != 2 or not np.isfinite(gain_range).all()
                or not 0 < gain_range[0] <= gain_range[1]):
            raise ValueError('gain_range must be a positive ordered pair')
        self.elastic_prob, self.elastic_alpha, self.elastic_sigma = elastic_prob, elastic_alpha, elastic_sigma
        self.speckle_prob, self.speckle_std = speckle_prob, speckle_std
        self.gain_prob, self.gain_range, self.attenuation_max = gain_prob, gain_range, attenuation_max
        self.bright_spot_prob, self.bright_spot_strength = bright_spot_prob, bright_spot_strength

    def __call__(self, image, mask):
        pixels = np.asarray(image, dtype=np.float32) / 255.
        target = np.asarray(mask, dtype=np.uint8)
        height, width = pixels.shape
        rows, columns = np.meshgrid(np.arange(height, dtype=np.float32),
                                    np.arange(width, dtype=np.float32), indexing='ij')
        if random.random() < self.elastic_prob:
            sigma = max(1., min(height, width)*self.elastic_sigma)
            amplitude = min(height, width)*self.elastic_alpha
            displacements = []
            for _ in range(2):
                noise = np.random.uniform(-1, 1, pixels.shape).astype(np.float32)
                smooth = gaussian_filter(noise, sigma, mode='reflect')
                displacements.append(smooth * (amplitude / max(float(np.abs(smooth).max()), 1e-6)))
            coordinates = (rows + displacements[0], columns + displacements[1])
            pixels = map_coordinates(pixels, coordinates, order=1, mode='reflect')
            target = map_coordinates(target, coordinates, order=0, mode='reflect')
        if random.random() < self.speckle_prob:
            noise = np.random.normal(0, self.speckle_std, pixels.shape).astype(np.float32)
            pixels = pixels * (1 + noise)
        if random.random() < self.gain_prob:
            gain = random.uniform(*self.gain_range)
            attenuation = random.uniform(0, self.attenuation_max)
            depth = np.linspace(0, 1, height, dtype=np.float32)[:, None]
            pixels = pixels * gain * np.exp(-attenuation*depth)
        if random.random() < self.bright_spot_prob:
            centre_row, centre_column = random.uniform(0, height-1), random.uniform(0, width-1)
            radius = max(1., min(height, width)*random.uniform(.02, .08))
            spot = np.exp(-((rows-centre_row)**2 + (columns-centre_column)**2)/(2*radius**2))
            pixels = pixels + random.uniform(0, self.bright_spot_strength)*spot
        pixels = np.rint(np.clip(pixels, 0, 1)*255).astype(np.uint8)
        return Image.fromarray(pixels), Image.fromarray(target)
