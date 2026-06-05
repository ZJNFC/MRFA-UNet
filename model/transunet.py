import torch
from torch import nn
import torch.nn.functional as F
import math

class two_conv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = two_conv3d(in_channels, out_channels)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, N, C = x.shape
        
        Q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, V)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.out_proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + res
        
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + res
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )
        self.conv = two_conv3d(out_channels * 2, out_channels)
    
    def forward(self, x1, x2):
        x = self.upsample(x1)
        if x.shape[-3:] != x2.shape[-3:]:
            # padding if necessary due to odd input sizes
            diff = [x2.size(i) - x.size(i) for i in range(2,5)]
            x = F.pad(x, [diff[2]//2, diff[2]-diff[2]//2,
                          diff[1]//2, diff[1]-diff[1]//2,
                          diff[0]//2, diff[0]-diff[0]//2])
        x = torch.cat([x, x2], dim=1)
        x = self.conv(x)
        return x

class TransUNet(nn.Module):
    """3D TransUNet with 4× down-sampling (input → 1/16) and symmetric 4× up."""
    def __init__(self, in_channels=1, num_classes=1,
                 features=[16, 32, 64, 128],  # keep unchanged as requested
                 hidden_size=256, num_layers=8, num_heads=4, mlp_dim=512):
        super().__init__()

        # ---------- Encoder ----------
        self.input_conv = two_conv3d(in_channels, features[0])           # x1  f=16
        self.down1 = down(features[0], features[1])                      # x2  f=32  /2
        self.down2 = down(features[1], features[2])                      # x3  f=64  /4
        self.down3 = down(features[2], features[3])                      # x4  f=128 /8
        self.down4 = down(features[3], features[3])                      # x5  f=128 /16 (keep channel 128)

        # ---------- Transformer ----------
        self.patch_embed = nn.Conv3d(features[3], hidden_size, kernel_size=1)  # token dim -> hidden
        self.position_embed = nn.Parameter(torch.zeros(1, 6500, hidden_size))  # length will be trimmed
        self.transformer = TransformerEncoder(hidden_size, num_layers, num_heads, mlp_dim)
        self.conv_project = nn.Conv3d(hidden_size, features[3], kernel_size=1)  # back to 128 ch

        # ---------- Decoder ----------
        self.up1 = Up(features[3], features[3])   # 128→128  (x5 up -> x4)
        self.up2 = Up(features[3], features[2])   # 128→64   (x4 up -> x3)
        self.up3 = Up(features[2], features[1])   # 64→32    (x3 up -> x2)
        self.up4 = Up(features[1], features[0])   # 32→16    (x2 up -> x1)

        self.output = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        x1 = self.input_conv(x)      # shape: 1
        x2 = self.down1(x1)          # 1/2
        x3 = self.down2(x2)          # 1/4
        x4 = self.down3(x3)          # 1/8
        x5 = self.down4(x4)          # 1/16

        # ----- Transformer -----
        B, _, D16, H16, W16 = x5.shape
        x_embed = self.patch_embed(x5)             # [B, hidden, D16, H16, W16]
        x_embed = x_embed.flatten(2).transpose(1, 2)  # [B, N, hidden]
        N = x_embed.size(1)
        x_embed = x_embed + self.position_embed[:, :N, :]
        x_trans = self.transformer(x_embed)
        x_trans = x_trans.transpose(1, 2).reshape(B, -1, D16, H16, W16)
        x_trans = self.conv_project(x_trans)       # [B, 128, D16, H16, W16]

        # ----- Decoder -----
        x = self.up1(x_trans, x4)   # ->128, 1/8
        x = self.up2(x,      x3)    # ->64,  1/4
        x = self.up3(x,      x2)    # ->32,  1/2
        x = self.up4(x,      x1)    # ->16,  1

        return self.output(x)
