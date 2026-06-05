import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv3d → BN → ReLU) × 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """DoubleConv → MaxPool3d (stride2)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.pool(x_conv)
        return x_conv, x_pool

class Up(nn.Module):
    """ConvTranspose3d 上采样 + 拼接(skip) + DoubleConv"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, x_skip):
        x = self.up(x)
        # padding if needed
        # if x.shape[-3:] != x_skip.shape[-3:]:
        #     diffZ = x_skip.size(2) - x.size(2)
        #     diffY = x_skip.size(3) - x.size(3)
        #     diffX = x_skip.size(4) - x.size(4)
        #     x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, features=(16,32,64,128)):
        super().__init__()
        f = features
        self.inc = DoubleConv(in_channels, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        # self.down4 = Down(f[3], f[4])
        # 底部不再池化，只用 DoubleConv
        self.bottom_conv = DoubleConv(f[3], f[3])

        # self.up1 = Up(f[4], f[4], f[3])  # in 256, skip256 → out128
        self.up2 = Up(f[3], f[3], f[2])  # 128, skip128 → 64
        self.up3 = Up(f[2], f[2], f[1])  # 64, skip64 → 32
        self.up4 = Up(f[1], f[1], f[0])  # 32, skip32 → 16
        self.outc = nn.Conv3d(f[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.inc(x)                # 16
        x1_skip, x = self.down1(x0)     # 32
        x2_skip, x = self.down2(x)      # 64
        x3_skip, x = self.down3(x)      # 128
        # x4_skip, x = self.down4(x)      # 256

        x = self.bottom_conv(x)         # 256 (same size as x4_skip after pool)

        # x = self.up1(x, x4_skip)        # 128
        x = self.up2(x, x3_skip)        # 64
        x = self.up3(x, x2_skip)        # 32
        x = self.up4(x, x1_skip)        # 16

        return self.outc(x)

# 旧别名
unet = UNet3D
