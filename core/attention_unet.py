import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules import EncoderBlockDropout, DoubleConvDropout, AttentionDecoderBlock

class AttentionUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, base_filters=64, dropout_prob=0.0):
        super(AttentionUNet, self).__init__()

        self.enc1 = EncoderBlockDropout(input_channels, base_filters, dropout_prob=dropout_prob)
        self.enc2 = EncoderBlockDropout(base_filters, base_filters * 2, dropout_prob=dropout_prob)
        self.enc3 = EncoderBlockDropout(base_filters * 2, base_filters * 4, dropout_prob=dropout_prob)
        self.enc4 = EncoderBlockDropout(base_filters * 4, base_filters * 8, dropout_prob=dropout_prob)

        self.bottleneck = DoubleConvDropout(base_filters * 8, base_filters * 16, dropout_prob=dropout_prob)

        self.dec1 = AttentionDecoderBlock(in_channels=base_filters * 16,
                                 out_channels=base_filters * 8,
                                 skip_channels=base_filters * 8,
                                 gate_channels=base_filters * 16,
                                 dropout_prob=dropout_prob)

        self.dec2 = AttentionDecoderBlock(in_channels=base_filters * 8,
                                 out_channels=base_filters * 4,
                                 skip_channels=base_filters * 4,
                                 gate_channels=base_filters * 8,
                                 dropout_prob=dropout_prob)

        self.dec3 = AttentionDecoderBlock(in_channels=base_filters * 4,
                                 out_channels=base_filters * 2,
                                 skip_channels=base_filters * 2,
                                 gate_channels=base_filters * 4,
                                 dropout_prob=dropout_prob)

        self.dec4 = AttentionDecoderBlock(in_channels=base_filters * 2,
                                 out_channels=base_filters,
                                 skip_channels=base_filters,
                                 gate_channels=base_filters * 2,
                                 dropout_prob=dropout_prob)

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
