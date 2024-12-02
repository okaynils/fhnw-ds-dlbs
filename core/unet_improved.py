import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetImproved(nn.Module):
    def __init__(self, num_classes):
        super(UNetImproved, self).__init__()
        
        # Encoder
        self.enc_conv1 = self.double_conv(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv4 = self.double_conv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = self.double_conv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(128, 64)
        
        # Final output layer
        self.final_layer = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def double_conv(self, in_channels, out_channels):
        """Creates a sequence of two convolutional layers with BatchNorm and ReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        x1 = self.enc_conv1(x)
        x1_pooled = self.pool1(x1)
        
        x2 = self.enc_conv2(x1_pooled)
        x2_pooled = self.pool2(x2)
        
        x3 = self.enc_conv3(x2_pooled)
        x3_pooled = self.pool3(x3)
        
        x4 = self.enc_conv4(x3_pooled)
        x4_pooled = self.pool4(x4)
        
        # Bottleneck
        x_bottleneck = self.bottleneck(x4_pooled)
        
        # Decoder path with skip connections
        x = self.upconv4(x_bottleneck)
        x = F.interpolate(x, size=x4.shape[2:], mode="bilinear", align_corners=False)  # Match x4 size
        x = torch.cat((x, x4), dim=1)
        x = self.dec_conv4(x)
        
        x = self.upconv3(x)
        x = F.interpolate(x, size=x3.shape[2:], mode="bilinear", align_corners=False)  # Match x3 size
        x = torch.cat((x, x3), dim=1)
        x = self.dec_conv3(x)
        
        x = self.upconv2(x)
        x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)  # Match x2 size
        x = torch.cat((x, x2), dim=1)
        x = self.dec_conv2(x)
        
        x = self.upconv1(x)
        x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)  # Match x1 size
        x = torch.cat((x, x1), dim=1)
        x = self.dec_conv1(x)
        
        # Final output layer
        output = self.final_layer(x)
        
        return output