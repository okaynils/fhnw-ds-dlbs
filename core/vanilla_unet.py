import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules import EncoderBlock, DoubleConv, DecoderBlock

class VanillaUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, base_filters=64):
        """
        Vanilla U-Net implementation in PyTorch.

        Args:
        - num_classes: Number of output classes for segmentation.
        - input_channels: Number of input image channels (default: 3 for RGB).
        - base_filters: Number of filters in the first layer (default: 64).
        """
        super(VanillaUNet, self).__init__()

        self.enc1 = EncoderBlock(input_channels, base_filters)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)

        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)

        self.dec1 = DecoderBlock(base_filters * 16, base_filters * 8)
        self.dec2 = DecoderBlock(base_filters * 8, base_filters * 4)
        self.dec3 = DecoderBlock(base_filters * 4, base_filters * 2)
        self.dec4 = DecoderBlock(base_filters * 2, base_filters)

        self.final_layer = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)

        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)

        outputs = self.final_layer(d4)
        return outputs