import torch
from torch import nn
from torch.nn import functional as F

class CCSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv3d(in_channels=2*channels, out_channels=channels,
                              kernel_size=1)
        self.act = nn.Hardsigmoid()
        self.bn = nn.BatchNorm3d(channels)

    def forward(self, x):
        mean = F.adaptive_avg_pool3d(x, output_size=1)                    # [B,C,1,1,1]
        sq_mean = F.adaptive_avg_pool3d(x * x, output_size=1)             # E[x^2]
        std = torch.sqrt(torch.clamp(sq_mean - mean * mean, min=1e-6))    # 稳健 std
        s = torch.cat([std, mean], dim=1)  # [B,2C,1,1,1]

        a = self.conv(s)
        a = self.act(a)
        x_ccsa = x*a
        x_ccsa = self.bn(x_ccsa)
        return x_ccsa

class CCSA_attention(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels,in_channels,kernel_size=3,padding=1),nn.BatchNorm3d(in_channels),nn.ReLU(inplace=True))
        self.channel_Attention = CCSA(out_channels)
        self.conv3 = nn.Sequential(nn.Conv3d(out_channels,out_channels,kernel_size=1),nn.BatchNorm3d(out_channels),nn.ReLU(inplace=True))
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.channel_Attention(x)
        x = self.conv3(x)
        return x

class MRF_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.DWconv = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.DWconv_group = nn.ModuleList([
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels)
        ])

    def forward(self,x):
        x = self.proj(x)
        x = self.DWconv(x)
        x_group = []
        for conv in self.DWconv_group:
            x_group.append(conv(x))
        return x_group

# 放在文件前部与其它模块并列
class ECA3D(nn.Module):

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    # x : [B,C,D,H,W]
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)   # [B,C,1]
        y = y.transpose(1, 2)                          # -> [B,1,C]
        y = self.conv(y)                               # in_channels = 1
        y = self.sigmoid(y).transpose(1, 2)            # -> [B,C,1]
        y = y.unsqueeze(-1).unsqueeze(-1)              # -> [B,C,1,1,1]
        return x * y          

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

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.pool(x_conv)
        return x_conv, x_pool

class Up(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch): 
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch+skip_ch, out_ch)

    def forward(self, x, x_skip):
        x = self.up(x)
        x = torch.cat([x_skip, x], dim=1)
        x = self.conv(x)
        return x 

class SAF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.eca = ECA3D(out_channels)
    def forward(self, x_dec,x_mrf):
        x_dec = self.conv(x_dec)
        x_dec_tilde = F.interpolate(x_dec, size=x_mrf.shape[2:], mode='trilinear', align_corners=False)
        x_fuse = x_dec_tilde + x_mrf
        x_saf = self.eca(x_fuse) 
        return x_saf

class MRFA_UNet(nn.Module):
    def __init__(self, in_channels, num_classes, features=(16,32,64,128)):
        super().__init__()
        f = features
        self.inc = DoubleConv(in_channels, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        self.bottom_conv = CCSA_attention(f[3], f[3])

        self.up1 = Up(f[3], f[3], f[2])  # 128, skip128 → 64
        self.up2 = Up(f[2], f[2], f[1])  # 64, skip64 → 32
        self.up3 = Up(f[1], f[1], f[0])  # 32, skip32 → 16

        self.outc = nn.Conv3d(f[0], num_classes, kernel_size=1)
        self.MRF = MRF_block(in_channels, f[0])
        
        self.SAF1 = SAF(f[3],f[0])
        self.SAF2 = SAF(f[2],f[0])
        self.SAF3 = SAF(f[1],f[0])


    def forward(self, x):

        x_mrf_3, x_mrf_5, x_mrf_7 = self.MRF(x)

        xenc0 = self.inc(x)
        x1_skip, xenc1 = self.down1(xenc0)
        x2_skip, xenc2 = self.down2(xenc1)
        x3_skip, xenc3 = self.down3(xenc2)

        # 最深层特征
        x3_dec = self.bottom_conv(xenc3)

        # 解码主干
        x2_dec = self.up1(x3_dec, x3_skip)
        x1_dec = self.up2(x2_dec, x2_skip)
        x0_dec = self.up3(x1_dec, x1_skip)

        x37_saf = self.SAF1(x3_dec, x_mrf_7)
        x25_saf = self.SAF2(x2_dec, x_mrf_5)
        x13_saf = self.SAF3(x1_dec, x_mrf_3)

        # 与最终浅层解码特征融合
        fusion = x0_dec + x13_saf + x25_saf + x37_saf
        return self.outc(fusion)
