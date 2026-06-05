import torch
from torch import nn
import torch.nn.functional as F

# -----------------------------------------------------
# Basic block: Conv3D → BN → ReLU
# -----------------------------------------------------
class conv3d_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# -----------------------------------------------------
# Encoder block: Conv → Conv → MaxPool with indices
# -----------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = conv3d_block(in_channels, out_channels)
        self.conv2 = conv3d_block(out_channels, out_channels)
        self.pool = nn.MaxPool3d(2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_pooled, indices = self.pool(x)
        return x_pooled, indices, x.shape  # send shape for unpooling

# -----------------------------------------------------
# Decoder block: MaxUnpool → Conv → Conv
# -----------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unpool = nn.MaxUnpool3d(2, stride=2)
        self.conv1 = conv3d_block(in_channels, out_channels)
        self.conv2 = conv3d_block(out_channels, out_channels)

    def forward(self, x, indices, output_shape):
        x = self.unpool(x, indices, output_size=output_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# -----------------------------------------------------
# 3D SegNet model
# -----------------------------------------------------
class SegNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, features=[16, 32, 64, 128]):
        super().__init__()

        # Encoder path
        self.enc1 = Encoder(in_channels, features[0])
        self.enc2 = Encoder(features[0], features[1])
        self.enc3 = Encoder(features[1], features[2])
        self.enc4 = Encoder(features[2], features[3])

        # Decoder path
        self.dec1 = Decoder(features[3], features[2])
        self.dec2 = Decoder(features[2], features[1])
        self.dec3 = Decoder(features[1], features[0])
        self.dec4 = Decoder(features[0], features[0])   # after first layer

        # Output layer
        self.out_conv = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # ---- Encoder ----
        x1, ind1, size1 = self.enc1(x)
        x2, ind2, size2 = self.enc2(x1)
        x3, ind3, size3 = self.enc3(x2)
        x4, ind4, size4 = self.enc4(x3)

        # ---- Decoder ----
        x = self.dec1(x4, ind4, size4)
        x = self.dec2(x, ind3, size3)
        x = self.dec3(x, ind2, size2)
        x = self.dec4(x, ind1, size1)

        # ---- Output ----
        x = self.out_conv(x)
        return x
