import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes, encoder_dims, decoder_dims):
        """
        Initialize the dynamic U-Net.
        
        Parameters:
        - encoder_dims: List of integers representing the output channels for each encoder layer.
        - decoder_dims: List of integers representing the output channels for each decoder layer.
        - num_classes: Number of output classes for semantic segmentation.
        """
        super(UNet, self).__init__()

        assert len(encoder_dims) == len(decoder_dims), "Encoder and decoder dimensions must match in length."

        self.num_layers = len(encoder_dims)
        
        self.enc_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_channels = 3
        for out_channels in encoder_dims:
            self.enc_layers.append(self.double_conv(in_channels, out_channels))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        self.bottleneck = self.double_conv(encoder_dims[-1], encoder_dims[-1] * 2)
        
        self.upconv_layers = nn.ModuleList()
        self.dec_layers = nn.ModuleList()
        
        in_channels = encoder_dims[-1] * 2
        for i, out_channels in enumerate(decoder_dims):
            self.upconv_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.dec_layers.append(self.double_conv(out_channels + encoder_dims[-(i + 1)], out_channels))
            in_channels = out_channels
        
        self.final_layer = nn.Conv2d(decoder_dims[-1], num_classes, kernel_size=1)
    
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
        enc_outputs = []
        
        for enc_layer, pool_layer in zip(self.enc_layers, self.pool_layers):
            x = enc_layer(x)
            enc_outputs.append(x)
            x = pool_layer(x)
        
        x = self.bottleneck(x)
        
        for i, (upconv, dec_layer) in enumerate(zip(self.upconv_layers, self.dec_layers)):
            x = upconv(x)
            skip_connection = enc_outputs[-(i + 1)]
            x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((x, skip_connection), dim=1)
            x = dec_layer(x)
        
        output = self.final_layer(x)
        return output
