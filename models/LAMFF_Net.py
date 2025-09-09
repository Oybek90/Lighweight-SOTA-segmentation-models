from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class DWconv(nn.Module):
    def __init__(self,dim_in,dim_out,kernel_size=3,padding=1,stride=1,dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in,dim_in,kernel_size=kernel_size,padding=padding,
                               stride=stride,dilation=dilation,groups=dim_in)
        self.noem_layer = nn.GroupNorm(4,dim_in)
        self.conv2 = nn.Conv2d(dim_in,dim_out,kernel_size=1,groups=dim_in)

    def forward(self,x):
        return self.conv2(self.noem_layer(self.conv1(x)))


class Gate_att(nn.Module):
    def __init__(self,in_c,out_c,kernel_size):
        super().__init__()
        self.w1 = nn.Sequential(
            DWconv(in_c,out_c,kernel_size,padding=kernel_size//2),
            nn.Sigmoid()
        )
        self.w2 = nn.Sequential(
            DWconv(in_c,in_c,kernel_size+2,padding=(kernel_size+2)//2),
            nn.GELU()
        )
        self.wo = nn.Sequential(
            DWconv(in_c,out_c,kernel_size),
            nn.GELU()
        )
        self.cw = nn.Conv2d(in_c,out_c,1,groups=in_c)

    def forward(self,x):
        x1, x2 = self.w1(x),self.w2(x)
        out = self.wo(x1*x2+self.cw(x))
        return out


class AMS(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.selayer0 = SEModule(dim)
        auto_padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=auto_padding, groups=dim, dilation=dilation)
        self.selayer = SEModule(dim)

        self.conv0_ = nn.Conv2d(dim//4,dim//4,3,padding=1,groups=dim//4)
        self.conv0_1 = nn.Conv2d(dim//4, dim//4, (1, 5), padding=(0, 2), groups=dim//4)
        self.conv0_2 = nn.Conv2d(dim//4, dim//4, (5, 1), padding=(2, 0), groups=dim//4)
        self.selayer1 = SEModule(dim//4,reduction=4)

        self.conv1_1 = nn.Conv2d(dim//4, dim//4, (1, 7), padding=(0, 3), groups=dim//4)
        self.conv1_2 = nn.Conv2d(dim//4, dim//4, (7, 1), padding=(3, 0), groups=dim//4)
        self.selayer2 = SEModule(dim//4,reduction=4)

        self.conv2_1 = nn.Conv2d(dim//4, dim//4, (1, 11), padding=(0, 5), groups=dim//4)
        self.conv2_2 = nn.Conv2d(dim//4, dim//4, (11, 1), padding=(5, 0), groups=dim//4)
        self.selayer3 = SEModule(dim//4,reduction=4)

        self.norm_layer = nn.GroupNorm(4,dim)
        self.conv2 = nn.Conv2d(dim,dim,1)
        self.gelu = nn.GELU()

        self.gau = Gate_att(dim,dim,3)

    def forward(self, x):
        attn = self.conv(x)

        attna = self.conv0(attn)
        se = self.selayer(attna)

        x = torch.chunk(se,4,dim=1)
        conv0_ = self.conv0_(x[0])
        attn_0 = self.conv0_1(x[1])
        attn_0 = self.conv0_2(attn_0)
        attn_0 = self.selayer1(attn_0)

        attn_1 = self.conv1_1(x[2])
        attn_1 = self.conv1_2(attn_1)
        attn_1 = self.selayer2(attn_1)

        attn_2 = self.conv2_1(x[3])
        attn_2 = self.conv2_2(attn_2)
        attn_2 = self.selayer3(attn_2)

        cat = torch.cat((conv0_,attn_0,attn_1,attn_2),dim=1)
        bn = self.norm_layer(cat)
        conv2 = self.conv2(bn)
        gelu = self.gelu(conv2)
        x = self.gau(gelu)

        return se+x




class Attention(nn.Module):
    def __init__(self, d_model, dilation=1):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv2d(d_model, d_model, 1)
        self.act = nn.GELU()
        self.ams = AMS(d_model, dilation=dilation)
        self.conv2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.ams(x)
        x = self.conv2(x)
        return x

class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = d_model * 4
        self.conv1 = nn.Conv2d(d_model, hidden_dim, 1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(hidden_dim, d_model, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x



class AMSE(nn.Module):
    def __init__(self, dim, dilation=1):
        super(AMSE, self).__init__()
        self.bn1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(dim)
        self.ffn = FFN(dim)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.bn1(x)
        x = self.attn(x)
        x = self.bn2(x)
        x = self.ffn(x)
        x = x + shortcut
        x = self.act(x)
        return x
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, layer_num=1, dilation=1):
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(AMSE(out_channels, dilation=dilation))
        super(Down, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.AvgPool2d(kernel_size=2,stride=2),
            #nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            *layers
        )

class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super(SpatialAttention, self).__init__()
        self.squeeze = nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(dim // reduction, dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SSE(nn.Module):
    def __init__(self, in_channels):
        super(SSE, self).__init__()
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.sSE(x)


# class SEModule(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEModule, self).__init__()
#
#         self.se_layer = SELayer(in_channels, reduction)
#
#     def forward(self, x):
#         # 将输入张量按像素分成四份
#         input_split = torch.split(x, split_size_or_sections=x.size(2) // 2 + 1, dim=2)
#         input_split = [torch.split(sub_tensor, split_size_or_sections=x.size(3) // 2 + 1, dim=3) for sub_tensor in input_split]
#         input_split = [item for sublist in input_split for item in sublist]  # 将列表展平
#
#         # 对每一份进行SE通道注意力
#         output_tensors = [self.se_layer(sub_tensor) for sub_tensor in input_split]
#
#         # 将四份结果合并
#         output_tensor_dim2 = torch.cat((output_tensors[0],output_tensors[1]), dim=3)
#         output_tensor_dim3 = torch.cat((output_tensors[2],output_tensors[3]), dim=3)
#         output_tensor = torch.cat((output_tensor_dim2,output_tensor_dim3), dim=2)
#
#         output_x = self.se_layer(x)
#
#         return output_tensor+output_x



def interpolate_to_match_resolution(tensor1, tensor2):
    # 获取两个张量的高度和宽度
    height1, width1 = tensor1.shape[-2], tensor1.shape[-1]
    height2, width2 = tensor2.shape[-2], tensor2.shape[-1]

    # 如果 tensor2 的分辨率小于 tensor1，执行插值操作
    if height2 < height1 or width2 < width1:
        # 计算目标分辨率
        target_height, target_width = height1, width1

        # 使用 F.interpolate 函数插值调整 tensor2
        tensor2 = F.interpolate(tensor2, size=(target_height, target_width), mode='bilinear', align_corners=False)

    return tensor2


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation =dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class Up4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up4, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels[0])
        self.norm2 = nn.BatchNorm2d(in_channels[1])

        self.scse1 = SEModule(in_channels[0])

        self.adjust_channel1 = nn.Conv2d(in_channels[1],in_channels[0],1,groups=16)
        self.scse2 = SEModule(in_channels[0])

        self.adjust_channel2 = nn.Conv2d(in_channels[0] * 2, 2, 1,groups=2)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.conv_last = ConvBnRelu(in_planes=in_channels[0], out_planes=in_channels[0], ksize=1, stride=1, pad=0,
                                    dilation=1)

        # self.conv3_0 = nn.Conv2d(in_channels[0],in_channels[0],3,padding=1,groups=16)
        # self.BN0 = nn.BatchNorm2d(in_channels[0])
        # self.RELU0 = nn.ReLU(inplace=True)

        self.conv3_1 = nn.Conv2d(in_channels[0]+in_channels[2], out_channels, 3, padding=1, groups=out_channels)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.RELU1 = nn.ReLU(inplace=True)

        #self.conv = nn.Sequential(AMSE(out_channels))

    def forward(self, x1, x2, x=None):
        if x != None:
            x = F.interpolate(input=x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
            diff_y = x1.size()[2] - x.size()[2]
            diff_x = x1.size()[3] - x.size()[3]

            # padding_left, padding_right, padding_top, padding_bottom
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        x1 = self.norm1(x1)
        x2 = self.norm2(x2)

        x1 = self.scse1(x1)

        x2 = F.interpolate(input=x2, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x2 = self.adjust_channel1(x2)
        x2 = self.scse2(x2)

        diff_y = x1.size()[2] - x2.size()[2]
        diff_x = x1.size()[3] - x2.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x2 = F.pad(x2, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        cat = torch.cat((x1,x2),dim=1)
        cat = self.adjust_channel2(cat)

        att = F.softmax(cat, dim=1)
        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)
        #att_3 = att[:, 2, :, :].unsqueeze(1)
        fusion = att_1 * x1 + att_2 * x2

        ax = self.relu(self.gamma * fusion + (1 - self.gamma) * (x1 + x2))
        ax = self.conv_last(ax)


        # cat = self.conv3_0(cat)
        # cat = self.BN0(cat)
        # cat = self.RELU0(cat)

        cat1 = torch.cat((x,ax),dim=1)

        cat1 = self.conv3_1(cat1)
        cat1 = self.BN1(cat1)
        out = self.RELU1(cat1)

        #out = self.conv(out)

        return out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels[0])
        self.norm2 = nn.BatchNorm2d(in_channels[1])
        self.norm3 = nn.BatchNorm2d(in_channels[2])

        #self.conv = nn.Sequential(AMSE(out_channels))

        self.pool0 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.adjust_channel0 = nn.Conv2d(in_channels[0],in_channels[1],1, groups=16)
        self.scse1 = SEModule(in_channels[1])

        self.scse2 = SEModule(in_channels[1])

        self.adjust_channel1 = nn.Conv2d(in_channels[2],in_channels[1],1,groups=16)
        self.scse3 = SEModule(in_channels[1])

        self.adjust_channel2 = nn.Conv2d(in_channels[1]*3,3,kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.conv_last = ConvBnRelu(in_planes=in_channels[1], out_planes=in_channels[1], ksize=1, stride=1, pad=0, dilation=1)

        # self.conv3_0 = nn.Conv2d(in_channels[1],in_channels[1],3,padding=1,groups=16)
        # self.BN0 = nn.BatchNorm2d(in_channels[1])
        # self.RELU0 = nn.ReLU(inplace=True)

        self.conv3_1 = nn.Conv2d(in_channels[1]+in_channels[3] if len(in_channels)==4 else in_channels[1]+in_channels[2], out_channels, 3, padding=1, groups=16)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.RELU1 = nn.ReLU(inplace=True)




    def forward(self, x1, x2, x3, x=None):
        if x != None:
            x = F.interpolate(input=x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
            diff_y = x2.size()[2] - x.size()[2]
            diff_x = x2.size()[3] - x.size()[3]

            # padding_left, padding_right, padding_top, padding_bottom
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        x3 = self.norm3(x3)

        x1 = self.pool0(x1)
        x1 = self.adjust_channel0(x1)
        x1 = self.scse1(x1)

        x2 = self.scse2(x2)

        x3 = F.interpolate(input=x3, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x3 = self.adjust_channel1(x3)
        x3 = self.scse3(x3)

        diff_y = x2.size()[2] - x3.size()[2]
        diff_x = x2.size()[3] - x3.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x3 = F.pad(x3, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])


        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        cat = torch.cat((x1,x2,x3),dim=1)
        cat = self.adjust_channel2(cat)

        att = F.softmax(cat, dim=1)
        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)
        att_3 = att[:, 2, :, :].unsqueeze(1)
        fusion = att_1 * x1 + att_2 * x2 + att_3 * x3

        ax = self.relu(self.gamma * fusion + (1 - self.gamma) * (x1+x2+x3))
        ax = self.conv_last(ax)

        # cat = self.conv3_0(ax)
        # cat = self.BN0(cat)
        # cat = self.RELU0(cat)

        cat1 = torch.cat((x,ax),dim=1) if x!=None else torch.cat((x3,ax),dim=1)

        cat1 = self.conv3_1(cat1)
        cat1 = self.BN1(cat1)
        out = self.RELU1(cat1)
        #out = self.conv(out)

        return out

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class LAMFFNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear=True,
                 base_c: int = 32):
        super(LAMFFNet, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_c, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(base_c),
            nn.GELU(),
            AMSE(base_c)
        )
        self.down1 = Down(base_c, base_c * 2, dilation=1)
        self.down2 = Down(base_c * 2, base_c * 4, dilation=1)
        self.down3 = Down(base_c * 4, base_c * 8, dilation=3)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16//factor, dilation=5)

        # c32
        self.up1 = Up(in_channels=[128,256,256], out_channels=base_c*4)
        self.up2 = Up(in_channels=[64,128,256,128], out_channels=base_c*2)
        self.up3 = Up(in_channels=[32,64,128,64], out_channels=base_c)
        self.up4 = Up4(in_channels=[32,64,32], out_channels=base_c//2)


        # #c16
        # self.up1 = Up(in_channels=[64, 128, 128], out_channels=base_c * 4)
        # self.up2 = Up(in_channels=[32, 64, 128, 64], out_channels=base_c * 2)
        # self.up3 = Up(in_channels=[16, 32, 64, 32], out_channels=base_c)
        # self.up4 = Up4(in_channels=[16, 32, 16], out_channels=base_c // 2)


        self.out_conv = OutConv(base_c//2, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # X5 X4
        x = self.up1(x3, x4, x5)


        # X5 X4 X3
        x = self.up2(x2, x3, x4,x)


        # X4 X3 X2
        x = self.up3(x1, x2, x3,x)


        # X3 X2 X1
        x = self.up4(x1, x2, x)
        logits = self.out_conv(x)
        return logits


if __name__ == '__main__':
    model = LAMFFNet(in_channels=3, num_classes=1, base_c=32).to('cpu')
    input = torch.randn(4, 3, 224, 224).to('cpu')
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
