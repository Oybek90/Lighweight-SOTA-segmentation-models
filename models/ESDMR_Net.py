
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ExpandSqueezeBlock(nn.Module):
    """Expand-Squeeze (ES) Block as described in Figure 2"""
    def __init__(self, in_channels, kernel_size=3):
        super(ExpandSqueezeBlock, self).__init__()
        
        self.relu = nn.ReLU()
        
        # Expansion using depthwise separable convolution with specified kernel size
        self.expand = DepthwiseSeparableConv(in_channels, in_channels, 
                                           kernel_size=kernel_size, 
                                           padding=kernel_size//2)
        
        # Squeeze using 1x1 convolution (identity mapping)
        self.squeeze = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        # ReLU activation first
        out = self.relu(x)
        
        # Expansion operation
        out = self.expand(out)
        
        # Squeeze operation
        out = self.squeeze(out)
        
        # Batch normalization
        out = self.bn(out)
        
        return out

class DualMultiscaleResidualBlock(nn.Module):
    """Dual Multiscale Residual (DMR) Block as described in Figure 3 and Equations 3-8"""
    def __init__(self, in_channels):
        super(DualMultiscaleResidualBlock, self).__init__()
        
        # First level convolutions (Equations 3-4)
        self.conv1_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv1_5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, bias=False)
        
        self.bn1_3x3 = nn.BatchNorm2d(in_channels)
        self.bn1_5x5 = nn.BatchNorm2d(in_channels)
        
        # Second level convolutions (Equations 5-6)
        self.conv2_3x3 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv2_5x5 = nn.Conv2d(in_channels*2, in_channels, kernel_size=5, padding=2, bias=False)
        
        self.bn2_3x3 = nn.BatchNorm2d(in_channels)
        self.bn2_5x5 = nn.BatchNorm2d(in_channels)
        
        # Final 1x1 convolution (Equation 7)
        self.conv_1x1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, bias=False)
        
        # Shortcut connection
        self.shortcut_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(in_channels)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Store input for final residual connection (Equation 8)
        residual = x
        
        # First level operations (Equations 3-4)
        f1_1 = self.bn1_3x3(self.conv1_3x3(x))  # f1_1
        f1_2 = self.bn1_5x5(self.conv1_5x5(x))  # f1_2
        
        # Dual-cross feature sharing (Equations 5-6)
        concat_1 = torch.cat([f1_1, f1_2], dim=1)  # {f1_1, f1_2}
        concat_2 = torch.cat([f1_2, f1_1], dim=1)  # {f1_2, f1_1}
        
        f1_3 = self.relu(self.bn2_3x3(self.conv2_3x3(concat_1)))  # f1_3
        f1_4 = self.relu(self.bn2_5x5(self.conv2_5x5(concat_2)))  # f1_4
        
        # Final concatenation and 1x1 convolution (Equation 7)
        final_concat = torch.cat([f1_4, f1_3], dim=1)  # {f1_4, f1_3}
        f_p = self.conv_1x1(final_concat)  # f_p
        
        # Shortcut connection processing
        shortcut = self.relu(self.shortcut_bn(self.shortcut_conv(residual)))
        
        # Final output with residual connection (Equation 8)
        out = f_p + shortcut
        
        return out

class EncoderBlock(nn.Module):
    """Encoder Block with ES blocks of different scales"""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        # Input projection with downsampling
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.input_conv = DepthwiseSeparableConv(in_channels, out_channels)
        self.input_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        # Identity convolution for channel adjustment
        self.identity_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        
        # Multiple ES blocks with different kernel sizes
        self.es_3x3 = ExpandSqueezeBlock(out_channels, kernel_size=3)
        self.es_5x5 = ExpandSqueezeBlock(out_channels, kernel_size=5)
        self.es_7x7 = ExpandSqueezeBlock(out_channels, kernel_size=7)
        
        # Pooling with 3x3 kernel and stride 1
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        # Fusion convolution for combining multiple scales
        self.fusion_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        # Store skip connection before downsampling
        skip_feature = x
        
        # Downsample and process input
        x = self.downsample(x)
        x = self.relu(self.input_bn(self.input_conv(x)))
        
        # Multi-scale ES block processing
        x_3x3 = self.es_3x3(x)
        x_5x5 = self.es_5x5(x)
        x_7x7 = self.es_7x7(x)
        x_pool = self.avg_pool(self.identity_conv(x))
        
        # Combine all scales
        combined = torch.cat([x_3x3, x_5x5, x_7x7, x_pool], dim=1)
        output = self.fusion_conv(combined)
        
        return output, skip_feature

class DecoderBlock(nn.Module):
    """Decoder Block with bilinear upsampling"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        # Bilinear upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(in_channels + skip_channels, out_channels, 
                                   kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Identity convolution for channel adjustment
        self.identity_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        
        # ES blocks with different scales
        self.es_3x3 = ExpandSqueezeBlock(out_channels, kernel_size=3)
        self.es_5x5 = ExpandSqueezeBlock(out_channels, kernel_size=5)
        self.es_7x7 = ExpandSqueezeBlock(out_channels, kernel_size=7)
        
        # Pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        # Final fusion convolution for combining multiple scales
        self.final_fusion_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x, skip_feature):
        # Upsample
        x = self.upsample(x)
        
        # Concatenate with skip feature
        x = torch.cat([x, skip_feature], dim=1)
        
        # Feature fusion
        x = self.relu(self.fusion_bn(self.fusion_conv(x)))
        
        # Multi-scale ES block processing
        x_3x3 = self.es_3x3(x)
        x_5x5 = self.es_5x5(x)
        x_7x7 = self.es_7x7(x)
        x_pool = self.avg_pool(self.identity_conv(x))
        
        # Combine all scales
        combined = torch.cat([x_3x3, x_5x5, x_7x7, x_pool], dim=1)
        output = self.final_fusion_conv(combined)
        
        return output

class ESRMNet(nn.Module):
    """ESDMR-Net: Expand-Squeeze Dual Multiscale Residual Network
    
    Lightweight network with approximately 0.7M parameters as reported in paper.
    """
    def __init__(self, in_channels=3, num_classes=1):
        super(ESRMNet, self).__init__()
        
        # Use smaller base channels to achieve ~0.7M parameters
        base_channels = 9
        
        # Input block
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        # Four encoder blocks
        self.encoder1 = EncoderBlock(base_channels, base_channels*2)      # 16
        self.encoder2 = EncoderBlock(base_channels*2, base_channels*4)    # 32
        self.encoder3 = EncoderBlock(base_channels*4, base_channels*6)    # 48
        self.encoder4 = EncoderBlock(base_channels*6, base_channels*8)    # 64
        
        # DMR blocks for skip connections
        self.dmr1 = DualMultiscaleResidualBlock(base_channels)      # 8
        self.dmr2 = DualMultiscaleResidualBlock(base_channels*2)    # 16
        self.dmr3 = DualMultiscaleResidualBlock(base_channels*4)    # 32
        self.dmr4 = DualMultiscaleResidualBlock(base_channels*6)    # 48
        
        # Four decoder blocks
        self.decoder4 = DecoderBlock(base_channels*8, base_channels*6, base_channels*6)  # 64->48
        self.decoder3 = DecoderBlock(base_channels*6, base_channels*4, base_channels*4)  # 48->32
        self.decoder2 = DecoderBlock(base_channels*4, base_channels*2, base_channels*2)  # 32->16
        self.decoder1 = DecoderBlock(base_channels*2, base_channels, base_channels)      # 16->8
        
        # Output block
        self.output_block = nn.Sequential(
            nn.Conv2d(base_channels, num_classes, kernel_size=1),
            # Softmax/sigmoid will be applied in loss function
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
    
    def forward(self, x):
        # Input processing
        x = self.input_block(x)
        
        # Encoder path
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, skip4 = self.encoder4(x3)
        
        # Apply DMR blocks to skip connections for feature preservation
        skip1_dmr = self.dmr1(skip1)
        skip2_dmr = self.dmr2(skip2)
        skip3_dmr = self.dmr3(skip3)
        skip4_dmr = self.dmr4(skip4)
        
        # Decoder path
        x = self.decoder4(x4, skip4_dmr)
        x = self.decoder3(x, skip3_dmr)
        x = self.decoder2(x, skip2_dmr)
        x = self.decoder1(x, skip1_dmr)
        
        # Output
        x = self.output_block(x)
        
        return x
