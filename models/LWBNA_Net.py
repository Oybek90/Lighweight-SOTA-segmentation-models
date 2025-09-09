import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class ModifiedSEBlock(nn.Module):
    """
    Modified Squeeze and Excitation Block with attention weights limited between 0.5 and 1.0
    as described in the paper
    """
    
    def __init__(self, channels, reduction=16):
        super(ModifiedSEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        # Global average pooling
        y = self.global_avg_pool(x).view(b, c)
        # Fully connected layers
        y = self.fc(y).view(b, c, 1, 1)
        
        # Modified activation: ReLU + Sigmoid to limit weights between 0.5 and 1.0
        attention_weights = 0.5 + 0.5 * torch.sigmoid(y)
        
        return x * attention_weights


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv3x3 -> BatchNorm -> ReLU
    """
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """
    Encoder block with attention
    """
    
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        self.attention = ModifiedSEBlock(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        # Convolutional operations
        x = self.conv_block(x)
        # Apply attention
        x = self.attention(x)
        # Keep skip connection before pooling
        skip = x
        # Pooling and dropout
        x = self.pool(x)
        x = self.dropout(x)
        
        return x, skip


class DecoderBlock(nn.Module):
    """
    Decoder block with attention and upsampling
    Based on paper's approach with add layers instead of concatenation
    """
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Since we're using add instead of concatenation, we need to match channels
        if in_channels != skip_channels:
            self.channel_match = nn.Conv2d(in_channels, skip_channels, kernel_size=1, bias=False)
        else:
            self.channel_match = None
            
        self.conv_block = nn.Sequential(
            ConvBlock(skip_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        self.attention = ModifiedSEBlock(out_channels)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x, skip):
        # Upsampling
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=False)
        
        # Match channels if needed
        if self.channel_match is not None:
            x = self.channel_match(x)
        
        # Add skip connection (as per paper's approach)
        x = x + skip
        
        # Convolutional operations
        x = self.conv_block(x)
        # Apply attention
        x = self.attention(x)
        # Dropout
        x = self.dropout(x)
        
        return x


class BottleneckNarrowing(nn.Module):
    """
    Bottleneck narrowing with attention (BNA) module
    Successive narrowing of channels from 128 to 16 in four steps
    """
    
    def __init__(self, in_channels=128):
        super(BottleneckNarrowing, self).__init__()
        
        # Successive narrowing: 128 -> 64 -> 32 -> 16
        self.narrow1 = nn.Sequential(
            ConvBlock(in_channels, in_channels // 2),
            ModifiedSEBlock(in_channels // 2)
        )
        
        self.narrow2 = nn.Sequential(
            ConvBlock(in_channels // 2, in_channels // 4),
            ModifiedSEBlock(in_channels // 4)
        )
        
        self.narrow3 = nn.Sequential(
            ConvBlock(in_channels // 4, in_channels // 8),
            ModifiedSEBlock(in_channels // 8)
        )
        
        # Skip connection and final attention
        self.skip_conv = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, bias=False)
        self.final_attention = ModifiedSEBlock(in_channels)
    
    def forward(self, x):
        identity = x
        
        # Successive narrowing
        x = self.narrow1(x)  # 128 -> 64
        x = self.narrow2(x)  # 64 -> 32
        x = self.narrow3(x)  # 32 -> 16
        
        # Expand back to original channels for skip connection
        x = self.skip_conv(x)  # 16 -> 128
        
        # Skip connection
        x = x + identity
        
        # Final attention
        x = self.final_attention(x)
        
        return x


class LWBNA_Unet(nn.Module):
    """
    Lightweight Bottleneck Narrowing with Attention U-Net
    Based on the paper's architecture with fixed 128 channels and bottleneck narrowing
    """
    
    def __init__(self, in_channels=3, num_classes=1, base_channels=128):
        super(LWBNA_Unet, self).__init__()
        
        # Initial convolution to match base_channels
        self.initial_conv = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, base_channels)
        )
        
        # Encoder path (all using fixed base_channels as per paper)
        self.encoder1 = EncoderBlock(base_channels, base_channels)
        self.encoder2 = EncoderBlock(base_channels, base_channels)
        self.encoder3 = EncoderBlock(base_channels, base_channels)
        
        # Bottleneck with narrowing and attention
        self.bottleneck_conv = ConvBlock(base_channels, base_channels)
        self.bottleneck_narrowing = BottleneckNarrowing(base_channels)
        
        # Decoder path - all using base_channels to match skip connections
        self.decoder3 = DecoderBlock(base_channels, base_channels, base_channels)
        self.decoder2 = DecoderBlock(base_channels, base_channels, base_channels)
        self.decoder1 = DecoderBlock(base_channels, base_channels, base_channels)
        
        # Final output layer
        self.final_conv = nn.Sequential(
            ConvBlock(base_channels, 64),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Store original size
        original_size = x.size()[2:]
        
        # Initial convolution to base_channels
        x = self.initial_conv(x)
        
        # Encoder path - all outputs will have base_channels
        x, skip1 = self.encoder1(x)  # base_channels -> base_channels
        x, skip2 = self.encoder2(x)  # base_channels -> base_channels
        x, skip3 = self.encoder3(x)  # base_channels -> base_channels
        
        # Bottleneck with narrowing
        x = self.bottleneck_conv(x)
        x = self.bottleneck_narrowing(x)
        
        # Decoder path - all using base_channels
        x = self.decoder3(x, skip3)  # base_channels + base_channels -> base_channels
        x = self.decoder2(x, skip2)  # base_channels + base_channels -> base_channels
        x = self.decoder1(x, skip1)  # base_channels + base_channels -> base_channels
        
        # Final output
        x = self.final_conv(x)
        
        # Resize to original input size if needed
        if x.size()[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        return x
