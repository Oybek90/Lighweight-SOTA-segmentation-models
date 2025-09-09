import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class IEA(nn.Module):
    """
    Inverted External Attention Module (Final Corrected Version)
    This version includes .contiguous() to prevent backward pass errors.
    """
    def __init__(self, in_channels, d_k=64):
        super().__init__()
        self.mem_unit_1 = nn.Linear(in_channels, d_k, bias=False)
        self.mem_unit_2 = nn.Linear(d_k, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(0, 2, 1)
        attn = self.mem_unit_1(x)
        attn = self.softmax(attn)
        attn = self.mem_unit_2(attn)

        # THE FIX: Add .contiguous() before .view()
        x = attn.permute(0, 2, 1).contiguous().view(b, c, h, w)

        return x + identity

class DECA(nn.Module):
    """
    Dilated Efficient Channel Attention Module (Final Corrected Version)
    """
    def __init__(self, in_channels, gamma=2, b=1):
        super().__init__()

        self.groups = 4
        assert in_channels % self.groups == 0, "Input channels must be divisible by the number of groups (4)"
        group_channels = in_channels // self.groups

        self.conv_d1 = self._depthwise_conv(group_channels, 1)
        self.conv_d2 = self._depthwise_conv(group_channels, 2)
        self.conv_d5 = self._depthwise_conv(group_channels, 5)
        self.conv_d7 = self._depthwise_conv(group_channels, 7)

        self.conv_final = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        t = int(abs((math.log(in_channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_eca = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def _depthwise_conv(self, ch, dilation_rate):
        return nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=dilation_rate, groups=ch, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU6()
        )

    def forward(self, x):
        identity = x
        chunks = torch.chunk(x, self.groups, dim=1)

        # THE FIX: Call .contiguous() on each chunk before the convolution
        out_d1 = self.conv_d1(chunks[0].contiguous())
        out_d2 = self.conv_d2(chunks[1].contiguous())
        out_d5 = self.conv_d5(chunks[2].contiguous())
        out_d7 = self.conv_d7(chunks[3].contiguous())

        x = torch.cat((out_d1, out_d2, out_d5, out_d7), dim=1)
        x = self.conv_final(x)

        y = self.avg_pool(x)
        y = self.conv_eca(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return identity * y.expand_as(identity)

class TransitionBlock(nn.Module):
    """A lighter block with a single convolution for transitioning between stages."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class AB(nn.Module):
    """
    Attention Bridge Module
    Fixed: replaced BatchNorm1d with LayerNorm to handle [B, C, 1] inputs.
    """
    def __init__(self, channels_list):
        super().__init__()
        full_channels = sum(channels_list)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv1d = nn.Conv1d(full_channels, full_channels, kernel_size=3, padding=1, bias=False)
        self.ln = nn.LayerNorm(full_channels)   # <-- FIXED
        self.relu = nn.ReLU(inplace=True)

        self.fcs = nn.ModuleList([nn.Linear(full_channels, channels) for channels in channels_list])
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        pooled_features = [self.gap(f) for f in features]
        concatenated = torch.cat(pooled_features, dim=1)  # [B, sum(C), 1, 1]

        b, c, _, _ = concatenated.shape
        concatenated = concatenated.view(b, c, 1)  # [B, sum(C), 1]

        fused = self.conv1d(concatenated).squeeze(-1)  # [B, sum(C)]
        fused = self.relu(self.ln(fused))              # <-- LayerNorm instead of BatchNorm

        attention_maps = [self.sigmoid(fc(fused)) for fc in self.fcs]

        attended_features = []
        for i in range(len(features)):
            att_map = attention_maps[i].view(b, -1, 1, 1)
            attended_features.append(features[i] + (features[i] * att_map))

        return attended_features

# --- LeaNet Model (Final Version Matching the Paper) ---
class LeaNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        channels = [8, 16, 24, 32, 48, 64]

        # Using the lighter TransitionBlock (single conv) for ALL standard stages
        self.enc1 = TransitionBlock(in_channels, channels[0])
        self.enc2 = TransitionBlock(channels[0], channels[1])
        self.enc3 = TransitionBlock(channels[1], channels[2])
        self.enc4 = nn.Sequential(IEA(channels[2]), DECA(channels[2]), TransitionBlock(channels[2], channels[3]))
        self.enc5 = nn.Sequential(IEA(channels[3]), DECA(channels[3]), TransitionBlock(channels[3], channels[4]))
        self.enc6 = nn.Sequential(IEA(channels[4]), DECA(channels[4]), TransitionBlock(channels[4], channels[5]))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ab = AB(channels_list=channels[:5])

        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec6 = TransitionBlock(channels[5] + channels[4], channels[4])
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec5 = TransitionBlock(channels[4] + channels[3], channels[3])
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = TransitionBlock(channels[3] + channels[2], channels[2])
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = TransitionBlock(channels[2] + channels[1], channels[1])
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = TransitionBlock(channels[1] + channels[0], channels[0])

        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass is unchanged
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))
        ab1, ab2, ab3, ab4, ab5 = self.ab([e1, e2, e3, e4, e5])
        d6 = self.dec6(torch.cat([self.up6(e6), ab5], dim=1))
        d5 = self.dec5(torch.cat([self.up5(d6), ab4], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), ab3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), ab2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), ab1], dim=1))
        output = self.out_conv(d2)
        # return self.sigmoid(output)
        return output


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

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    model = LeaNet(in_channels=3, out_channels=1)
    x = torch.randn(4, 3, 224, 224)  # input images
    y = torch.randint(0, 2, (4, 1, 224, 224)).float()  # ground truth masks

    preds = model(x)
    print("Output shape:", preds.shape)

    # Loss
    criterion = BCEDiceLoss()
    loss = criterion(preds, y)
    print("Loss:", loss.item())

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable: {trainable_params}")

