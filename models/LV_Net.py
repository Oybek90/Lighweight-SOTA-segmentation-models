import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.layers.weight_init as weight_init
from torchvision import models

# -------- SAFE BATCHNORM (prevents crash with batch_size==1 or H=W=1) --------
class SafeBatchNorm2d(nn.BatchNorm2d):
    """
    Behaves like BatchNorm2d, but if training with an input that would cause
    BN to error (e.g., batch size == 1 or spatial 1x1), it falls back to
    using running stats (i.e., eval-mode BN for that forward pass only).
    """
    def forward(self, x):
        if self.training:
            N, C, H, W = x.shape
            # BN needs >1 value per channel -> if N==1 and H*W==1, crash.
            # Also guard common failure when N==1 regardless of H,W.
            if N == 1 or (H * W) == 1:
                return F.batch_norm(
                    x,
                    self.running_mean,
                    self.running_var,
                    self.weight,
                    self.bias,
                    False,       # use running stats
                    0.0,         # momentum ignored when training=False
                    self.eps
                )
        return super().forward(x)

# -------------------------------- Activation --------------------------------
class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.deploy = deploy
        # depthwise conv kernel (2*act_num+1)
        self.weight = nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bias = None
        # Use SafeBatchNorm2d instead of plain BN
        self.bn = SafeBatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num
        weight_init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            # Padding matches kernel size to keep shape
            return F.conv2d(
                super().forward(x),
                self.weight,
                self.bias,
                padding=(self.act_num * 2 + 1) // 2,
                groups=self.dim
            )
        # training path: depthwise conv -> SafeBN
        return self.bn(
            F.conv2d(
                super().forward(x),
                self.weight,
                padding=self.act_num,
                groups=self.dim
            )
        )

    def _fuse_bn_tensor(self, weight, bn: SafeBatchNorm2d):
        # Fuse BN into conv using running stats (as deploy will)
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        # bias term for deploy fusion
        return weight * t, bn.bias - bn.running_mean * (bn.weight / std)

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = nn.Parameter(bias)
        self.__delattr__('bn')
        self.deploy = True

# --------------------------------- Block ------------------------------------
class Block(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=2, deploy=False):
        super().__init__()
        self.act_learn = 0.0
        self.deploy = deploy
        if deploy:
            self.conv = nn.Conv2d(dim, dim_out, 1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, 1, bias=False),
                SafeBatchNorm2d(dim, eps=1e-6)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, 1, bias=False),
                SafeBatchNorm2d(dim_out, eps=1e-6)
            )
        self.pool = nn.MaxPool2d(stride) if stride > 1 else nn.Identity()
        self.act = activation(dim_out, act_num, deploy)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            x = F.leaky_relu(x, self.act_learn)
            x = self.conv2(x)
        return self.act(self.pool(x))

    def _fuse_bn_tensor(self, conv: nn.Conv2d, bn: SafeBatchNorm2d):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        bias = conv.bias if conv.bias is not None else 0
        # fuse conv+bn
        fused_weight = conv.weight * t
        fused_bias = bn.bias - (bn.running_mean - bias) * (bn.weight / std)
        return fused_weight, fused_bias

    def switch_to_deploy(self):
        if self.deploy:
            return
        # fuse conv1 + bn1
        k1, b1 = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        # fuse conv2 + bn2
        k2, b2 = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])

        # Combine two 1x1 convs into a single 1x1 conv:
        # new_weight = k2 @ k1 (channel-wise)
        in_ch = self.conv1[0].in_channels
        mid_ch = self.conv1[0].out_channels
        out_ch = self.conv2[0].out_channels

        # reshape for matmul
        k1_flat = k1.view(mid_ch, in_ch)          # [mid, in]
        k2_flat = k2.view(out_ch, mid_ch)         # [out, mid]
        k_comb = torch.matmul(k2_flat, k1_flat)   # [out, in]
        k_comb = k_comb.view(out_ch, in_ch, 1, 1)

        b_comb = b2 + torch.matmul(k2_flat, b1.view(-1, 1)).view(-1)

        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        with torch.no_grad():
            self.conv.weight.copy_(k_comb)
            self.conv.bias.copy_(b_comb)

        # cleanup
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True

# -------------------------------- UpBlock -----------------------------------
class UpBlock(Block):
    def __init__(self, dim, dim_out, act_num=3, factor=2, deploy=False):
        super().__init__(dim, dim_out, act_num, 1, deploy)
        self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)
        self.__delattr__('pool')

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            x = F.leaky_relu(x, self.act_learn)
            x = self.conv2(x)
        return self.act(self.upsample(x))

# -------------------------------- LV_UNet -----------------------------------
class LV_UNet(nn.Module):
    def __init__(self, deploy=False):
        super().__init__()
        self.deploy = deploy
        mobile = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

        # Encoder from MobileNetV3
        self.firstconv = mobile.features[0]
        self.encoder1 = nn.Sequential(mobile.features[1], mobile.features[2])
        self.encoder2 = nn.Sequential(mobile.features[3], mobile.features[4], mobile.features[5])
        self.encoder3 = nn.Sequential(mobile.features[6], mobile.features[7], mobile.features[8], mobile.features[9])

        # Channel dims observed in MobileNetV3-Large at these cuts
        dims = [80, 160, 240, 480]
        self.stages = nn.ModuleList([
            Block(dims[i], dims[i + 1], stride=2, deploy=deploy) for i in range(len(dims) - 1)
        ])

        # Decoder stage 1 (fusable path)
        self.up_stages1 = nn.ModuleList([
            UpBlock(dims[3], dims[2], deploy=deploy),  # 480 -> 240
            UpBlock(dims[2], dims[1], deploy=deploy),  # 240 -> 160
            UpBlock(dims[1], dims[0], deploy=deploy)   # 160 -> 80
        ])

        # Skip adapters for fusable path
        self.skip_conv1 = nn.Conv2d(240, 240, 1)  # s2 skip
        self.skip_conv2 = nn.Conv2d(160, 160, 1)  # s1 skip
        self.skip_conv3 = nn.Conv2d(80, 80, 1)    # x3 skip

        # Decoder for pretrained features
        self.up_stages2 = nn.ModuleList([
            UpBlock(80, 40, deploy=deploy),
            UpBlock(40, 24, deploy=deploy),
            UpBlock(24, 16, deploy=deploy)
        ])

        # Skip adapters for pretrained features
        self.skip_conv4 = nn.Conv2d(40, 40, 1)    # x2 skip
        self.skip_conv5 = nn.Conv2d(24, 24, 1)    # x1 skip
        self.skip_conv6 = nn.Conv2d(16, 16, 1)    # x0 skip

        self.final_up = UpBlock(16, 16, deploy=deploy)
        self.final_conv = nn.Conv2d(16, 1, 1)

    def change_act(self, m):
        for module in self.modules():
            if hasattr(module, 'act_learn'):
                module.act_learn = m

    def forward(self, x):
        # Pre-trained Encoder
        x0 = self.firstconv(x)      # shallow
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)      # deepest from backbone

        # Fusable Encoder
        s1 = self.stages[0](x3)
        s2 = self.stages[1](s1)
        s3 = self.stages[2](s2)

        # Decoder stage 1 with skips
        d1 = self.up_stages1[0](s3)
        s2_resized = F.interpolate(s2, size=d1.shape[2:], mode='bilinear', align_corners=False) if d1.shape[2:] != s2.shape[2:] else s2
        d1 = d1 + self.skip_conv1(s2_resized)

        d2 = self.up_stages1[1](d1)
        s1_resized = F.interpolate(s1, size=d2.shape[2:], mode='bilinear', align_corners=False) if d2.shape[2:] != s1.shape[2:] else s1
        d2 = d2 + self.skip_conv2(s1_resized)

        d3 = self.up_stages1[2](d2)
        x3_resized = F.interpolate(x3, size=d3.shape[2:], mode='bilinear', align_corners=False) if d3.shape[2:] != x3.shape[2:] else x3
        d3 = d3 + self.skip_conv3(x3_resized)

        # Decoder stage 2 with pretrained feature skips
        d4 = self.up_stages2[0](d3)
        x2_resized = F.interpolate(x2, size=d4.shape[2:], mode='bilinear', align_corners=False) if d4.shape[2:] != x2.shape[2:] else x2
        d4 = d4 + self.skip_conv4(x2_resized)

        d5 = self.up_stages2[1](d4)
        x1_resized = F.interpolate(x1, size=d5.shape[2:], mode='bilinear', align_corners=False) if d5.shape[2:] != x1.shape[2:] else x1
        d5 = d5 + self.skip_conv5(x1_resized)

        d6 = self.up_stages2[2](d5)
        x0_resized = F.interpolate(x0, size=d6.shape[2:], mode='bilinear', align_corners=False) if d6.shape[2:] != x0.shape[2:] else x0
        d6 = d6 + self.skip_conv6(x0_resized)

        out = self.final_up(d6)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.final_conv(out)

    def switch_to_deploy(self):
        for module in self.modules():
            if hasattr(module, 'switch_to_deploy') and module is not self:
                module.switch_to_deploy()
        self.deploy = True
