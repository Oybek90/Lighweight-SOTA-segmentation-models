import torch
import torch.nn as nn
import torch.nn.functional as F

# =================== MODEL BUILDING BLOCKS ===================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class MultiDownsamplingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        spatial_down = self.pool(x)
        freq_spec = torch.fft.fft2(x)
        shifted_spec = torch.fft.fftshift(freq_spec, dim=(-2, -1))
        skip_connection = shifted_spec
        _, _, h, w = shifted_spec.shape
        h_start, w_start = h // 4, w // 4
        h_end, w_end = h_start + h // 2, w_start + w // 2
        cropped_spec = shifted_spec[:, :, h_start:h_end, w_start:w_end]
        unshifted_spec = torch.fft.ifftshift(cropped_spec, dim=(-2, -1))
        freq_down = torch.fft.ifft2(unshifted_spec)
        downsampled_output = spatial_down + freq_down.real
        return downsampled_output, skip_connection


class SpectrumUpsamplingBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, skip_spec):
        low_res_spec = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        padded_spec = torch.zeros_like(skip_spec)
        _, _, h, w = skip_spec.shape
        _, _, h_low, w_low = low_res_spec.shape
        h_start, w_start = (h - h_low) // 2, (w - w_low) // 2
        h_end, w_end = h_start + h_low, w_start + w_low
        padded_spec[:, :, h_start:h_end, w_start:w_end] = low_res_spec
        mask = torch.ones_like(skip_spec)
        mask[:, :, h_start:h_end, w_start:w_end] = 0
        high_freq_spec = skip_spec * mask
        fused_spec = padded_spec + high_freq_spec
        upsampled_img = torch.fft.ifft2(torch.fft.ifftshift(fused_spec, dim=(-2, -1)))
        return upsampled_img.real



class TokenizedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=2):
        super().__init__()
        # The dimension to which channels are expanded
        expanded_channels = in_channels * expansion_factor

        self.norm = nn.GroupNorm(1, in_channels)

        # New MLP block with channel expansion and contraction
        self.mlp = nn.Sequential(
            # 1. 1x1 Conv to expand channels
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1),
            nn.GELU(),
            # 2. 3x3 Depthwise Conv on expanded channels
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, padding=1, groups=expanded_channels,
                      bias=False),
            nn.GELU(),
            # 3. 1x1 Conv to contract channels
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1),
        )

        self.residual_conv = nn.Conv2d(in_channels, out_channels,
                                       kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        _, _, h, w = x.shape
        residual = self.residual_conv(x)
        x_shifted_h = torch.roll(x, shifts=(h // 2), dims=2)
        x_shifted_v = torch.roll(x, shifts=(w // 2), dims=3)

        x_processed = self.norm(x_shifted_h + x_shifted_v)
        processed = self.mlp(x_processed)

        output = residual + processed
        return output


# =================== FULL MLU-NET MODEL ===================
class MLUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        channels = [16, 32, 128, 160, 256]
        self.in_conv = DoubleConv(in_channels, channels[0])
        self.enc_mdb1 = MultiDownsamplingBlock()
        self.enc_conv2 = DoubleConv(channels[0], channels[1])
        self.enc_mdb2 = MultiDownsamplingBlock()
        self.enc_conv3 = DoubleConv(channels[1], channels[2])
        self.enc_mdb3 = MultiDownsamplingBlock()
        self.pool = nn.MaxPool2d(2)
        self.enc_mlp4 = TokenizedMLP(channels[2], channels[3])
        self.enc_mlp5 = TokenizedMLP(channels[3], channels[4])
        self.dec_mlp5 = TokenizedMLP(channels[4], channels[3])
        self.dec_mlp4 = TokenizedMLP(channels[3], channels[2])
        self.dec_sub3 = SpectrumUpsamplingBlock()
        self.dec_conv3 = DoubleConv(channels[2] + channels[2], channels[1])
        self.dec_sub2 = SpectrumUpsamplingBlock()
        self.dec_conv2 = DoubleConv(channels[1] + channels[1], channels[0])
        self.dec_sub1 = SpectrumUpsamplingBlock()
        self.dec_conv1 = DoubleConv(channels[0] + channels[0], channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.in_conv(x)
        d1, skip1_spec = self.enc_mdb1(s1)
        s2 = self.enc_conv2(d1)
        d2, skip2_spec = self.enc_mdb2(s2)
        s3 = self.enc_conv3(d2)
        d3, skip3_spec = self.enc_mdb3(s3)
        s4 = self.pool(s3)
        mlp4_out = self.enc_mlp4(s4)
        s5 = self.pool(mlp4_out)
        bottleneck = self.enc_mlp5(s5)
        u5 = nn.functional.interpolate(bottleneck, scale_factor=2)
        u5 = self.dec_mlp5(u5)
        u5 = u5 + mlp4_out
        u4 = nn.functional.interpolate(u5, scale_factor=2)
        u4 = self.dec_mlp4(u4)
        u3 = self.dec_sub3(u4, skip3_spec)
        u3 = torch.cat([u3, s3], dim=1)
        u3 = self.dec_conv3(u3)
        u2 = self.dec_sub2(u3, skip2_spec)
        u2 = torch.cat([u2, s2], dim=1)
        u2 = self.dec_conv2(u2)
        u1 = self.dec_sub1(u2, skip1_spec)
        u1 = torch.cat([u1, s1], dim=1)
        u1 = self.dec_conv1(u1)
        return self.out_conv(u1)
