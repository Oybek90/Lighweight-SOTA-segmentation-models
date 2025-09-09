import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channel dim for [B,C,H,W]."""
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D [B,C,H,W], got {x.shape}")
        x = x.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# MobileNetV2-style MBConv (expansion -> DWConv -> SE -> PW)
class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand=4):
        super().__init__()
        mid = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.expand = nn.Sequential(
            ConvBNAct(in_ch, mid, k=1, s=1, p=0)
        ) if expand != 1 else nn.Identity()
        self.dw = ConvBNAct(mid, mid, k=3, s=stride, g=mid)  # depthwise
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, max(1, mid // 8), 1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(1, mid // 8), mid, 1, bias=True),
            nn.Sigmoid()
        )
        self.pw = ConvBNAct(mid, out_ch, k=1, s=1, p=0, act=False)
    def forward(self, x):
        y = self.expand(x) if isinstance(self.expand, nn.Sequential) else x
        y = self.dw(y)
        y = y * self.se(y)
        y = self.pw(y)
        return x + y if self.use_res else y

# -----------------------------
# Axial Attention (linear wrt H/W)
# -----------------------------
class AxialSelfAttention(nn.Module):
    """
    Axial self-attention along H then W.
    Input/Output: [B, C, H, W]. Heads split the channel dim; attention along sequence length.
    """
    def __init__(self, dim, heads=2):
        super().__init__()
        assert dim % heads == 0, "channels must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        # 1D projections for H-axis and W-axis sequences
        self.qkv_h = nn.Conv1d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj_h = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

        self.qkv_w = nn.Conv1d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj_w = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def _axis_attn(self, seq, qkv, proj):
        """
        seq: [B*, C, L]  (B* = batch times the other spatial axis)
        returns: [B*, C, L]
        """
        Bstar, C, L = seq.shape
        q, k, v = qkv(seq).chunk(3, dim=1)               # [B*, C, L] each
        H = self.heads
        c = self.head_dim

        q = q.view(Bstar, H, c, L)
        k = k.view(Bstar, H, c, L)
        v = v.view(Bstar, H, c, L)

        attn = torch.softmax((q.transpose(-2, -1) @ k) / (c ** 0.5), dim=-1)  # [B*, H, L, L]
        y = (attn @ v.transpose(-2, -1)).transpose(-2, -1)                    # [B*, H, c, L]
        y = y.reshape(Bstar, C, L)                                            # [B*, C, L]
        return proj(y)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        y = x

        # ---- H-axis: treat columns (fixed w) as length-H sequences
        h_seq = y.permute(0, 3, 1, 2).reshape(B * W, C, H)      # [B*W, C, H]
        h_out = self._axis_attn(h_seq, self.qkv_h, self.proj_h) # [B*W, C, H]
        h_out = h_out.reshape(B, W, C, H).permute(0, 2, 3, 1)   # -> [B, C, H, W]
        y = y + h_out

        # ---- W-axis: treat rows (fixed h) as length-W sequences
        w_seq = y.permute(0, 2, 1, 3).reshape(B * H, C, W)      # [B*H, C, W]
        w_out = self._axis_attn(w_seq, self.qkv_w, self.proj_w) # [B*H, C, W]
        w_out = w_out.reshape(B, H, C, W).permute(0, 2, 1, 3)   # -> [B, C, H, W]
        y = y + w_out

        return y

# -----------------------------
# MBA: Global (Axial) + Local (DW) + lightweight multi-scale fusion
# -----------------------------
class MBA(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.global_branch = nn.Sequential(
            ChannelLayerNorm(ch),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.GELU(),
            AxialSelfAttention(ch, heads=2),
            ConvBNAct(ch, ch, k=1, s=1, p=0)
        )
        self.local_branch = nn.Sequential(
            ConvBNAct(ch, ch, k=3, s=1, g=ch),  # depthwise
            ConvBNAct(ch, ch, k=1, s=1, p=0)    # pointwise
        )
        # Multi-scale channel aggregator
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool4 = nn.AdaptiveAvgPool2d(4)
        self.fuse = nn.Conv2d(ch * 3, ch, 1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(1, ch // 8), 1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(1, ch // 8), ch, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        g = self.global_branch(x)
        l = self.local_branch(x)
        u = (g + l) * 0.5
        p2 = F.interpolate(self.pool2(u), size=u.shape[-2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.pool4(u), size=u.shape[-2:], mode='bilinear', align_corners=False)
        m = torch.cat([u, p2, p4], dim=1)
        m = self.fuse(m)
        m = m * self.gate(m)
        return x + m  # residual

# -----------------------------
# LSM: Lightweight Shift-MLP block
# -----------------------------
class Shift(nn.Module):
    """Channel-wise axial shifting (up/down/left/right & identity)."""
    def __init__(self, ch):
        super().__init__()
        # split channels into 5 groups
        self.splits = [ch // 5] * 5
        self.splits[0] += ch - sum(self.splits)

    def forward(self, x):
        B, C, H, W = x.shape
        c0, c1, c2, c3, c4 = torch.split(x, self.splits, dim=1)
        def roll(t, dy, dx): return torch.roll(t, shifts=(dy, dx), dims=(2, 3))
        y0 = c0                           # identity
        y1 = roll(c1, -1, 0)              # up
        y2 = roll(c2,  1, 0)              # down
        y3 = roll(c3,  0, -1)             # left
        y4 = roll(c4,  0,  1)             # right
        return torch.cat([y0, y1, y2, y3, y4], dim=1)

class LSMBlock(nn.Module):
    def __init__(self, ch, mlp_ratio=4):
        super().__init__()
        self.norm1 = ChannelLayerNorm(ch)
        self.shift = Shift(ch)
        self.pw1   = nn.Conv2d(ch, ch, 1, bias=False)
        self.norm2 = ChannelLayerNorm(ch)
        hidden = int(ch * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, ch, 1, bias=False)
        )
    def forward(self, x):
        # token mixer
        y = self.shift(x)
        y = self.pw1(y)
        x = x + y
        # channel mixer (per-pixel MLP) — use ChannelLayerNorm directly
        y = self.norm2(x)
        y = self.mlp(y)
        return x + y

# -----------------------------
# FIS: Feature Information Share for skip connections
# -----------------------------
class FIS(nn.Module):
    """
    Spatial attention (shared dilated conv on [avg|max]) +
    Channel attention (GAP -> concat across stages -> 1D conv -> per-stage FC).
    """
    def __init__(self, stages_channels=(16, 32, 64, 128)):
        super().__init__()
        # Spatial attention: k=7, dilation=3 => effective kernel = 19 -> padding=9 to keep H/W
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, dilation=3, padding=9, bias=True)

        # Channel attention heads per stage (paper: 16,32,64,128)
        fc_sizes = [16, 32, 64, 128]
        total_c = sum(stages_channels)
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_c, fc_sizes[i], bias=True),
                nn.GELU(),
                nn.Linear(fc_sizes[i], c, bias=True),
                nn.Sigmoid()
            ) for i, c in enumerate(stages_channels)
        ])
        self.conv1d = nn.Conv1d(total_c, total_c, kernel_size=1, bias=True)

    def forward(self, skips):
        # skips: list [s1,s2,s3,s4] (low->high)
        s_attended = []
        for s in skips:
            avg = torch.mean(s, dim=1, keepdim=True)
            mx  = torch.amax(s, dim=1, keepdim=True)
            a = torch.cat([avg, mx], dim=1)            # [B,2,H,W]
            a = torch.sigmoid(self.spatial(a))         # [B,1,H,W]
            s_attended.append(s * a + s)               # residual

        # Channel attention across stages
        gaps = [F.adaptive_avg_pool2d(s, 1).flatten(1) for s in s_attended]  # [B,Ci]
        cat = torch.cat(gaps, dim=1).unsqueeze(-1)                            # [B,sumC,1]
        cat = self.conv1d(cat).squeeze(-1)                                    # [B,sumC]

        outs = []
        for i, s in enumerate(s_attended):
            gate = self.fc[i](cat)[:, :, None, None]  # [B,Ci,1,1]
            outs.append(s * gate + s)
        return outs

# -----------------------------
# Decoder Conv2d Block (bilinear upsample + Conv2d Block)
# -----------------------------
class Conv2dBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3),
            ConvBNAct(out_ch, out_ch, 3)
        )
    def forward(self, x):
        return self.block(x)

# -----------------------------
# LTMSegNet
# -----------------------------
class LTMSegNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=(16,32,64,128), num_classes=1):
        super().__init__()
        c1, c2, c3, c4 = base_ch

        # Encoder
        self.stem = nn.Sequential(
            MBConv(in_channels, c1, stride=2, expand=1),  # 1/2
            MBConv(c1, c1, stride=1, expand=4)
        )
        self.down2 = MBConv(c1, c2, stride=2, expand=4)  # 1/4
        self.mba2  = MBA(c2)
        self.down3 = MBConv(c2, c3, stride=2, expand=4)  # 1/8
        self.mba3  = MBA(c3)
        self.down4 = MBConv(c3, c4, stride=2, expand=4)  # 1/16
        self.lsm4  = LSMBlock(c4)
        self.bottleneck = LSMBlock(c4)

        # FIS on skips [s1,s2,s3,s4]
        self.fis = FIS(stages_channels=(c1, c2, c3, c4))

        # Decoder
        self.up4 = Conv2dBlock(c4 + c4, c4)            # bottleneck + skip4
        self.up3 = Conv2dBlock(c4 // 2 + c3, c3)
        self.up2 = Conv2dBlock(c3 // 2 + c2, c2)
        self.up1 = Conv2dBlock(c2 // 2 + c1, c1)

        self.reduce4 = nn.Conv2d(c4, c4 // 2, 1, bias=False)
        self.reduce3 = nn.Conv2d(c3, c3 // 2, 1, bias=False)
        self.reduce2 = nn.Conv2d(c2, c2 // 2, 1, bias=False)

        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x):
        # Encoder
        s1 = self.stem(x)                 # 1/2,  c1
        s2 = self.mba2(self.down2(s1))    # 1/4,  c2
        s3 = self.mba3(self.down3(s2))    # 1/8,  c3
        s4 = self.lsm4(self.down4(s3))    # 1/16, c4
        b  = self.bottleneck(s4)          # 1/16, c4

        # FIS-adjusted skips
        s1, s2, s3, s4 = self.fis([s1, s2, s3, s4])

        # Decoder
        y = torch.cat([b, s4], dim=1)
        y = self.up4(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)  # -> 1/8

        y = torch.cat([self.reduce4(y), s3], dim=1)
        y = self.up3(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)  # -> 1/4

        y = torch.cat([self.reduce3(y), s2], dim=1)
        y = self.up2(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)  # -> 1/2

        y = torch.cat([self.reduce2(y), s1], dim=1)
        y = self.up1(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)  # -> 1/1

        logits = self.head(y)
        return logits

