import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution: (Conv2D -> BatchNorm -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """
    Encoder block: DoubleConv -> MaxPool2D
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        downsampled = self.pool(skip)
        return skip, downsampled


class DecoderBlock(nn.Module):
    """
    Decoder block: TransposeConv2D -> Concatenate -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)  # Ensure correct shape
        x = torch.cat((x, skip), dim=1)  # Concatenate skip connection
        return self.conv(x)
    

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