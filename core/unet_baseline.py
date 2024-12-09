import torch
import torch.nn as nn

class UNetBaseline(nn.Module):
    def __init__(self, num_classes):
        super(UNetBaseline, self).__init__()
        
        # Encoder
        self.enc_conv1 = self.double_conv(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self.double_conv(128, 256)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(128, 64)
        
        self.final_layer = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def double_conv(self, in_channels, out_channels):
        """Creates a sequence of two convolutional layers with ReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        x1 = self.enc_conv1(x)
        x1_pooled = self.pool1(x1)
        
        x2 = self.enc_conv2(x1_pooled)
        x2_pooled = self.pool2(x2)
        
        # Bottleneck
        x_bottleneck = self.bottleneck(x2_pooled)
        
        # Decoder path with skip connections
        x = self.upconv2(x_bottleneck)
        x = torch.cat((x, x2), dim=1)
        x = self.dec_conv2(x)
        
        x = self.upconv1(x)
        x = torch.cat((x, x1), dim=1)
        x = self.dec_conv1(x)
        
        # Final output layer
        output = self.final_layer(x)
        
        return output
