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
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)
    

class EncoderBlockDropout(nn.Module):
    """
    Encoder block: DoubleConv -> MaxPool2D
    """
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(EncoderBlockDropout, self).__init__()
        self.conv = DoubleConvDropout(in_channels, out_channels, dropout_prob=dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        downsampled = self.pool(skip)
        return skip, downsampled

    

class DoubleConvDropout(nn.Module):
    """
    Double convolution: (Conv2D -> BatchNorm -> ReLU -> Dropout?) * 2
    """
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(DoubleConvDropout, self).__init__()
        self.dropout_prob = dropout_prob

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_prob),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_prob),
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
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
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionDecoderBlock(nn.Module):
    """
    Decoder block: (UpConv -> Attention -> Concat) -> DoubleConv
    """
    def __init__(self, in_channels, out_channels, skip_channels, gate_channels, dropout_prob=0.0):
        super(AttentionDecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionBlock(F_g=gate_channels, F_l=skip_channels, F_int=skip_channels // 2)
        self.conv = DoubleConvDropout(out_channels * 2, out_channels, dropout_prob=dropout_prob)

    def forward(self, x, skip, g):
        x = self.upconv(x)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        skip = self.attention(skip, g)

        x = torch.cat((x, skip), dim=1)
        return self.conv(x)
