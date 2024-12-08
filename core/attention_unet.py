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


class AttentionBlock(nn.Module):
    """
    Attention block: Given gating signal (g) and skip connection (x),
    this block computes an attention coefficient to filter x.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: skip connection (from encoder), shape [B, F_l, H, W]
        # g: gating signal (from decoder), shape [B, F_g, H', W']

        # Project gating and skip to F_int
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Align shapes (if necessary, via interpolation)
        # Ensure g1 and x1 spatial dims match
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)

        # Compute attention coefficients
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Multiply skip connection by attention map
        return x * psi


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, gate_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionBlock(F_g=gate_channels, F_l=skip_channels, F_int=skip_channels // 2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip, g):
        x = self.upconv(x)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        # Apply attention gating on skip connection
        skip = self.attention(skip, g)

        # Concatenate attended skip and upsampled feature
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, base_filters=64):
        super(AttentionUNet, self).__init__()

        # Encoder
        self.enc1 = EncoderBlock(input_channels, base_filters)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)

        # Decoder
        # dec1: in_channels=base_filters*16, out_channels=base_filters*8
        # Skip from enc4 has base_filters*8 channels (s4)
        # Gating from bottleneck has base_filters*16 channels (b)
        self.dec1 = DecoderBlock(in_channels=base_filters * 16,
                                 out_channels=base_filters * 8,
                                 skip_channels=base_filters * 8,
                                 gate_channels=base_filters * 16)

        # dec2: in_channels=base_filters*8, out_channels=base_filters*4
        # Skip from enc3 has base_filters*4 channels (s3)
        # Gating from dec1 has base_filters*8 channels (d1)
        self.dec2 = DecoderBlock(in_channels=base_filters * 8,
                                 out_channels=base_filters * 4,
                                 skip_channels=base_filters * 4,
                                 gate_channels=base_filters * 8)

        # dec3: in_channels=base_filters*4, out_channels=base_filters*2
        # Skip from enc2 has base_filters*2 channels (s2)
        # Gating from dec2 has base_filters*4 channels (d2)
        self.dec3 = DecoderBlock(in_channels=base_filters * 4,
                                 out_channels=base_filters * 2,
                                 skip_channels=base_filters * 2,
                                 gate_channels=base_filters * 4)

        # dec4: in_channels=base_filters*2, out_channels=base_filters
        # Skip from enc1 has base_filters channels (s1)
        # Gating from dec3 has base_filters*2 channels (d3)
        self.dec4 = DecoderBlock(in_channels=base_filters * 2,
                                 out_channels=base_filters,
                                 skip_channels=base_filters,
                                 gate_channels=base_filters * 2)

        # Final output layer
        self.final_layer = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)

        d1 = self.dec1(b, s4, g=b)
        d2 = self.dec2(d1, s3, g=d1)
        d3 = self.dec3(d2, s2, g=d2)
        d4 = self.dec4(d3, s1, g=d3)

        outputs = self.final_layer(d4)
        return outputs
