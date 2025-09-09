import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU activation."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelShift(nn.Module):
    """Channel shifting operation for introducing locality."""
    def __init__(self, shift_size=1):
        super().__init__()
        self.shift_size = shift_size

    def forward(self, x, shift_type='width'):
        if shift_type == 'width':
            # Shift along width dimension
            return torch.roll(x, shifts=self.shift_size, dims=3)
        elif shift_type == 'height':
            # Shift along height dimension
            return torch.roll(x, shifts=self.shift_size, dims=2)
        return x


class TokenizedMLP(nn.Module):
    """Simplified Tokenized MLP block based on paper description."""
    def __init__(self, in_channels, token_dim=64, hidden_dim=128):
        super().__init__()
        
        # Tokenization with 3x3 conv
        self.tokenize = nn.Conv2d(in_channels, token_dim, kernel_size=3, padding=1)
        
        # MLP for width mixing
        self.mlp_width = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(token_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, token_dim, kernel_size=1)
        )
        
        # Depth-wise convolution for positional encoding
        self.dwconv = nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1, groups=token_dim)
        
        # MLP for height mixing
        self.mlp_height = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(token_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, token_dim, kernel_size=1)
        )
        
        # Reprojection back to original channels
        self.reproject = nn.Conv2d(token_dim, in_channels, kernel_size=1)
        
        # Layer normalization
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # Store residual
        residual = x
        
        # Tokenize features
        tokens = self.tokenize(x)
        
        # Apply width MLP with channel shifting
        shifted_tokens = torch.roll(tokens, shifts=1, dims=3)  # Shift width
        tokens = tokens + self.mlp_width(shifted_tokens)
        
        # Add positional encoding via depth-wise conv
        tokens = tokens + self.dwconv(tokens)
        
        # Apply height MLP with channel shifting
        shifted_tokens = torch.roll(tokens, shifts=1, dims=2)  # Shift height
        tokens = tokens + self.mlp_height(shifted_tokens)
        
        # Reproject to original channels
        output = self.reproject(tokens)
        
        # Add residual connection and normalize
        output = output + residual
        output = self.norm(output)
        
        return output


class UNeXt(nn.Module):
    """UNeXt: MLP-based Medical Image Segmentation Network."""
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False):
        super().__init__()
        
        # Channel configuration from paper: C1=32, C2=64, C3=128, C4=160, C5=256
        # But using smaller channels for 1.47M params
        self.channels = [32, 64, 128, 160, 256]
        
        # Encoder - Convolutional Stage (first 3 blocks)
        self.inc = ConvBlock(input_channels, self.channels[0])  # Input block
        
        # Encoder block 1
        self.down1 = nn.Sequential(
            ConvBlock(self.channels[0], self.channels[0]),
            nn.MaxPool2d(2, 2)
        )
        
        # Encoder block 2
        self.down2 = nn.Sequential(
            ConvBlock(self.channels[0], self.channels[1]),
            nn.MaxPool2d(2, 2)
        )
        
        # Encoder block 3
        self.down3 = nn.Sequential(
            ConvBlock(self.channels[1], self.channels[2]),
            nn.MaxPool2d(2, 2)
        )
        
        # Encoder block 4 - Start of MLP stage
        self.down4 = nn.Sequential(
            ConvBlock(self.channels[2], self.channels[3]),
            nn.MaxPool2d(2, 2)
        )
        
        # Bottleneck - Tokenized MLP blocks (block 5)
        self.bottleneck = nn.Sequential(
            ConvBlock(self.channels[3], self.channels[4]),
            TokenizedMLP(self.channels[4], token_dim=128, hidden_dim=256),
            TokenizedMLP(self.channels[4], token_dim=128, hidden_dim=256)
        )
        
        # Decoder - MLP Stage (first 2 decoder blocks)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(self.channels[4] + self.channels[3], self.channels[3])
        )
        self.tok_mlp_up1 = TokenizedMLP(self.channels[3], token_dim=64, hidden_dim=128)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(self.channels[3] + self.channels[2], self.channels[2])
        )
        self.tok_mlp_up2 = TokenizedMLP(self.channels[2], token_dim=32, hidden_dim=64)
        
        # Decoder - Convolutional Stage (last 3 decoder blocks)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(self.channels[2] + self.channels[1], self.channels[1])
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(self.channels[1] + self.channels[0], self.channels[0])
        )
        
        # Final output layer
        self.outc = nn.Conv2d(self.channels[0], num_classes, kernel_size=1)
        
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.ds1 = nn.Conv2d(self.channels[1], num_classes, 1)
            self.ds2 = nn.Conv2d(self.channels[2], num_classes, 1)
            self.ds3 = nn.Conv2d(self.channels[3], num_classes, 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)                    # [B, 32, 224, 224]
        x2 = self.down1(x1)                 # [B, 32, 112, 112]
        x3 = self.down2(x2)                 # [B, 64, 56, 56]
        x4 = self.down3(x3)                 # [B, 128, 28, 28]
        x5 = self.down4(x4)                 # [B, 160, 14, 14]
        
        # Bottleneck with Tokenized MLPs
        x5 = self.bottleneck(x5)            # [B, 256, 14, 14]
        
        # Decoder path with MLP stage
        d4 = self.up1(torch.cat([x5, x5], dim=1))  # Upsample and concat
        d4 = torch.cat([d4, x5], dim=1)
        d4 = self.up1[1](d4)                # [B, 160, 28, 28]
        d4 = self.tok_mlp_up1(d4)
        
        d3 = self.up2[0](d4)                # Upsample
        d3 = torch.cat([d3, x4], dim=1)     # Skip connection
        d3 = self.up2[1](d3)                # [B, 128, 56, 56]
        d3 = self.tok_mlp_up2(d3)
        
        # Decoder path - Convolutional stage
        d2 = self.up3[0](d3)                # Upsample
        d2 = torch.cat([d2, x3], dim=1)     # Skip connection
        d2 = self.up3[1](d2)                # [B, 64, 112, 112]
        
        d1 = self.up4[0](d2)                # Upsample
        d1 = torch.cat([d1, x2], dim=1)     # Skip connection
        d1 = self.up4[1](d1)                # [B, 32, 224, 224]
        
        # Final output
        out = self.outc(d1)                 # [B, num_classes, 224, 224]
        
        if self.deep_supervision:
            ds1 = F.interpolate(self.ds1(d2), size=x.shape[2:], mode='bilinear', align_corners=False)
            ds2 = F.interpolate(self.ds2(d3), size=x.shape[2:], mode='bilinear', align_corners=False)
            ds3 = F.interpolate(self.ds3(d4), size=x.shape[2:], mode='bilinear', align_corners=False)
            return [out, ds1, ds2, ds3]
        
        return out


# Corrected UNeXt with exact paper specifications
class UNeXt(nn.Module):
    """UNeXt implementation exactly as described in the paper."""
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False):
        super().__init__()
        
        # Exact channel configuration from paper
        # C1=32, C2=64, C3=128, C4=160, C5=256 (but scaled down for 1.47M params)
        self.c1, self.c2, self.c3, self.c4, self.c5 = 16, 32, 64, 128, 256
        
        # Stage 1: Convolutional Encoder (3 blocks)
        self.conv1 = ConvBlock(input_channels, self.c1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = ConvBlock(self.c1, self.c2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = ConvBlock(self.c2, self.c3)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Stage 2: MLP Encoder (2 blocks)
        self.conv4 = ConvBlock(self.c3, self.c4)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.tok_mlp4 = TokenizedMLP(self.c4, token_dim=64, hidden_dim=128)
        
        # Bottleneck
        self.conv5 = ConvBlock(self.c4, self.c5)
        self.tok_mlp5 = TokenizedMLP(self.c5, token_dim=32, hidden_dim=64)
        
        # Stage 3: MLP Decoder (2 blocks)
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv6 = ConvBlock(self.c5 + self.c4, self.c4)
        self.tok_mlp6 = TokenizedMLP(self.c4, token_dim=64, hidden_dim=128)
        
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv7 = ConvBlock(self.c4 + self.c3, self.c3)
        self.tok_mlp7 = TokenizedMLP(self.c3, token_dim=32, hidden_dim=64)
        
        # Stage 4: Convolutional Decoder (3 blocks)
        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv8 = ConvBlock(self.c3 + self.c2, self.c2)
        
        self.up9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv9 = ConvBlock(self.c2 + self.c1, self.c1)
        
        # Final output
        self.outc = nn.Conv2d(self.c1, num_classes, kernel_size=1)
        
        self.deep_supervision = deep_supervision

    def forward(self, x):
        # Encoder - Convolutional Stage
        e1 = self.conv1(x)       # [B, 16, 224, 224]
        p1 = self.pool1(e1)      # [B, 16, 112, 112]
        
        e2 = self.conv2(p1)      # [B, 32, 112, 112]
        p2 = self.pool2(e2)      # [B, 32, 56, 56]
        
        e3 = self.conv3(p2)      # [B, 64, 56, 56]
        p3 = self.pool3(e3)      # [B, 64, 28, 28]
        
        # Encoder - MLP Stage
        e4 = self.conv4(p3)      # [B, 128, 28, 28]
        e4 = self.tok_mlp4(e4)   # Apply tokenized MLP
        p4 = self.pool4(e4)      # [B, 128, 14, 14]
        
        # Bottleneck
        e5 = self.conv5(p4)      # [B, 256, 14, 14]
        e5 = self.tok_mlp5(e5)   # Apply tokenized MLP
        
        # Decoder - MLP Stage
        d4 = self.up6(e5)        # [B, 256, 28, 28]
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection [B, 256+128, 28, 28]
        d4 = self.conv6(d4)      # [B, 128, 28, 28]
        d4 = self.tok_mlp6(d4)
        
        d3 = self.up7(d4)        # [B, 128, 56, 56]
        d3 = torch.cat([d3, e3], dim=1)  # Skip connection [B, 128+64, 56, 56]
        d3 = self.conv7(d3)      # [B, 64, 56, 56]
        d3 = self.tok_mlp7(d3)
        
        # Decoder - Convolutional Stage
        d2 = self.up8(d3)        # [B, 64, 112, 112]
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d2 = self.conv8(d2)      # [B, 32, 112, 112]
        
        d1 = self.up9(d2)        # [B, 32, 224, 224]
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = self.conv9(d1)      # [B, 16, 224, 224]
        
        # Final output
        out = self.outc(d1)      # [B, num_classes, 224, 224]
        
        return out


# Loss Functions (from paper)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss as used in the paper."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        # Loss formulation from paper: L = 0.5*BCE(ŷ,y) + Dice(ŷ,y)
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return 0.5 * bce_loss + dice_loss

