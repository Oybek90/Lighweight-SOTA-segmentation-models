import torch
import torch.nn as nn
import torch.nn.functional as F

# Cell 3: Standard Convolutional Block
class ConvBlock(nn.Module):
    """A standard 2D Convolutional Block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class UpConv(nn.Module):
    """The Upsampling Block for the decoder path."""
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_block = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels),
            ConvBlock(in_channels=out_channels, out_channels=out_channels)
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        return self.conv_block(x)
    
class LocalPMFSBlock(nn.Module):
    """LocalPMFSBlock optimized for 128-channel inputs."""
    def __init__(self, in_channels, out_channels):
        super(LocalPMFSBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels // 4, kernel_size=1, padding=0)
        self.conv3 = ConvBlock(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5 = ConvBlock(in_channels, out_channels // 2, kernel_size=5, padding=2)
        self.fusion = ConvBlock(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        return self.fusion(torch.cat([x1, x3, x5], dim=1))
    
class GlobalPMFSBlock(nn.Module):
    """GlobalPMFSBlock optimized for 128-channel inputs."""
    def __init__(self, in_channels, out_channels):
        super(GlobalPMFSBlock, self).__init__()
        self.d_conv1 = ConvBlock(in_channels, out_channels // 4, kernel_size=3, padding=1, dilation=1)
        self.d_conv3 = ConvBlock(in_channels, out_channels // 4, kernel_size=3, padding=3, dilation=3)
        self.d_conv5 = ConvBlock(in_channels, out_channels // 2, kernel_size=3, padding=5, dilation=5)
        self.fusion = ConvBlock(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.d_conv1(x)
        x3 = self.d_conv3(x)
        x5 = self.d_conv5(x)
        return self.fusion(torch.cat([x1, x3, x5], dim=1))
    
class PMFSNet(nn.Module):
    """
    The definitive PMFSNet 'BASIC' architecture (~951k parameters) that
    matches the research paper's specifications.
    """
    def __init__(self, in_channels=3, n_classes=1):
        super(PMFSNet, self).__init__()
        # This filter configuration creates the ~951k parameter BASIC model
        f = [8, 16, 32, 64, 128]

        # --- Encoder Path ---
        self.conv1 = nn.Sequential(ConvBlock(in_channels, f[0]), ConvBlock(f[0], f[0]))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(ConvBlock(f[0], f[1]), ConvBlock(f[1], f[1]))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(ConvBlock(f[1], f[2]), ConvBlock(f[2], f[2]))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(ConvBlock(f[2], f[3]), ConvBlock(f[3], f[3]))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Bottleneck ---
        self.bridge = nn.Sequential(ConvBlock(f[3], f[4]), ConvBlock(f[4], f[4]))
        self.local_pmfs = LocalPMFSBlock(f[4], f[4])
        self.global_pmfs = GlobalPMFSBlock(f[4], f[4])
        self.fusion = ConvBlock(f[4] * 2, f[4], kernel_size=1, padding=0)

        # --- Decoder Path ---
        self.up4 = UpConv(in_channels=f[4] + f[3], out_channels=f[3])
        self.up3 = UpConv(in_channels=f[3] + f[2], out_channels=f[2])
        self.up2 = UpConv(in_channels=f[2] + f[1], out_channels=f[1])
        self.up1 = UpConv(in_channels=f[1] + f[0], out_channels=f[0])

        # --- Output Layer ---
        self.out_conv = nn.Conv2d(f[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x); p1 = self.pool1(x1)
        x2 = self.conv2(p1); p2 = self.pool2(x2)
        x3 = self.conv3(p2); p3 = self.pool3(x3)
        x4 = self.conv4(p3); p4 = self.pool4(x4)
        bridge = self.bridge(p4)
        local_features = self.local_pmfs(bridge)
        global_features = self.global_pmfs(bridge)
        fused_features = self.fusion(torch.cat([local_features, global_features], dim=1))
        d4 = self.up4(fused_features, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        return self.out_conv(d1)
    
class DiceLoss(nn.Module):
    """
    Calculates the Dice Loss, a common metric for image segmentation tasks.
    The loss is calculated as 1 - Dice Coefficient.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # The model outputs raw logits, so we apply a sigmoid to get probabilities (0-1)
        inputs = torch.sigmoid(inputs)
        
        # Flatten the inputs and targets to 1D tensors to calculate the dot product
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate the intersection and the total elements
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    model = PMFSNet(in_channels=3, n_classes=1)
    x = torch.randn(4, 3, 224, 224)  # input images
    y = torch.randint(0, 2, (4, 1, 224, 224)).float()  # ground truth masks

    preds = model(x)
    print("Output shape:", preds.shape)

    # Loss
    criterion = DiceLoss()
    loss = criterion(preds, y)
    print("Loss:", loss.item())

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable: {trainable_params}")

