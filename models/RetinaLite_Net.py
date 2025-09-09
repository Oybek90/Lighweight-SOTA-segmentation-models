import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V):
        # Q, K, V shape: (batch_size, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output = self.scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # Final linear transformation
        output = self.W_o(attention_output)
        return output

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution Block.
    This is more efficient than a standard convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

class ConvBlock(nn.Module):
    """
    Updated Encoder block to use the efficient DepthwiseSeparableConv.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.dw_sep_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.dw_sep_conv(x)
        x = self.maxpool(x)
        return x

class DecoderBlock(nn.Module):
    """
    Updated Decoder block (without skip connection) to use efficient convolutions.
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = DepthwiseSeparableConv(out_channels, out_channels, 3, padding=1)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.upconv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderBlockWithSkip(nn.Module):
    """
    Updated Decoder block (with skip connection) to use efficient convolutions.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlockWithSkip, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.cbam = CBAM(skip_channels)
        self.conv1 = DepthwiseSeparableConv(out_channels + skip_channels, out_channels, 3, padding=1)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, 3, padding=1)

    def forward(self, x, skip):
        x = self.upconv(x)
        skip = self.cbam(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class RetinaLiteNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, num_heads=4, d_k=32):
        super(RetinaLiteNet, self).__init__()
        
        # Encoder
        self.conv1 = ConvBlock(in_channels, 32)      # 224x224 -> 112x112
        self.conv2 = ConvBlock(32, 64)               # 112x112 -> 56x56
        self.conv3 = ConvBlock(64, 128)              # 56x56 -> 28x28
        
        # Multi-head attention
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_model = 128
        self.mha = MultiHeadAttention(self.d_model, num_heads)
        
        # Decoder
        # Bottleneck: 256 channels (128 CNN + 128 MHA)
        self.decoder1 = DecoderBlockWithSkip(256, 64, 128)   # 28x28 -> 56x56, skip from conv2
        self.decoder2 = DecoderBlockWithSkip(128, 32, 64)    # 56x56 -> 112x112, skip from conv1
        self.decoder3 = DecoderBlock(64, 32)                 # 112x112 -> 224x224, no skip
        
        # Output layer
        self.final_conv = nn.Conv2d(32, num_classes, 1)
        
    def forward(self, x):
        # Debug prints to track dimensions
        # print(f"Input: {x.shape}")
        
        # Encoder
        x1 = self.conv1(x)      # [B, 32, 112, 112]
        # print(f"After conv1: {x1.shape}")
        
        x2 = self.conv2(x1)     # [B, 64, 56, 56]
        # print(f"After conv2: {x2.shape}")
        
        x3 = self.conv3(x2)     # [B, 128, 28, 28]
        # print(f"After conv3: {x3.shape}")
        
        # Multi-head attention
        B, C, H, W = x3.shape
        # Reshape for attention: [B, H*W, C]
        x3_reshaped = x3.view(B, C, H*W).transpose(1, 2)  # [B, H*W, 128]
        
        # Apply multi-head attention
        mha_out = self.mha(x3_reshaped)  # [B, H*W, 128]
        
        # Reshape back to feature map
        mha_out = mha_out.transpose(1, 2).view(B, C, H, W)  # [B, 128, 28, 28]
        
        # Feature fusion: concatenate CNN features and MHA features
        fused_features = torch.cat([x3, mha_out], dim=1)  # [B, 256, 28, 28]
        # print(f"Fused features: {fused_features.shape}")
        
        # Decoder with skip connections
        d1 = self.decoder1(fused_features, x2)  # [B, 128, 56, 56] with skip [B, 64, 56, 56]
        # print(f"After decoder1: {d1.shape}")
        
        d2 = self.decoder2(d1, x1)              # [B, 64, 112, 112] with skip [B, 32, 112, 112]
        # print(f"After decoder2: {d2.shape}")
        
        d3 = self.decoder3(d2)                  # [B, 32, 224, 224]
        # print(f"After decoder3: {d3.shape}")
        
        # Final output
        output = self.final_conv(d3)            # [B, 1, 224, 224]
        # print(f"Final output: {output.shape}")
        
        return output


