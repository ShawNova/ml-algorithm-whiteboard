import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Contracting Path (Encoder)
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512)
        )

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Expanding Path (Decoder)
        self.decoder = nn.Sequential(
            self.upconv_block(1024, 512),
            self.upconv_block(512, 256),
            self.upconv_block(256, 128),
            self.upconv_block(128, 64)
        )

        # Final Segmentation Output
        self.final_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Double Convolution Block (Conv2D -> ReLU -> Conv2D -> ReLU)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """Upsampling Block with Transposed Convolution"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder (with Skip Connections)
        dec1 = self.decoder[0](bottleneck) + enc4  # Skip Connection
        dec2 = self.decoder[1](dec1) + enc3
        dec3 = self.decoder[2](dec2) + enc2
        dec4 = self.decoder[3](dec3) + enc1

        # Final Segmentation Output
        output = self.final_layer(dec4)
        return output

# Create U-Net Model
model = UNet(in_channels=3, out_channels=1)
print(model)
