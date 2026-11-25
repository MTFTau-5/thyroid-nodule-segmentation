from torch import nn


class SegmentationHeads(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.coarse = nn.Conv2d(channels, 1, 1)
        self.segmentation = nn.Conv2d(channels, 1, 1)
        self.boundary = nn.Conv2d(channels, 1, 1)
