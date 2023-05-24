import torch
import torch.nn as nn
import torch.nn.functional as F

from models.switchable_norm import SwitchNorm2d

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding,
                                   groups=in_channels, dilation=dilation)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BAMBlock(nn.Module):
    def __init__(self, channels, reduction=32):
        super(BAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x).squeeze(3).squeeze(2)
        y = self.channel_attention(y).unsqueeze(2).unsqueeze(3)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bam=True, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.sn1 = SwitchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.sn2 = SwitchNorm2d(out_channels)
        self.conv3 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.sn3 = SwitchNorm2d(out_channels)
        self.bam = BAMBlock(out_channels) if use_bam else None

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                DepthwiseSeparableConv(in_channels, out_channels, kernel_size=1, padding=0, dilation=1),
                SwitchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        x = F.relu(self.sn1(self.conv1(x)))
        x = F.relu(self.sn2(self.conv2(x)))
        x = self.sn3(self.conv3(x))
        if self.bam:
            x = self.bam(x)
        x += self.shortcut(residual)
        x = F.relu(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, num_classes=10):
        super(SqueezeExcitation, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, 64, kernel_size=3, padding=1)
        self.sn1 = SwitchNorm2d(64)
        self.res_block1 = ResidualBlock(64, 64, dilation=2)
        self.res_block2 = ResidualBlock(64, 128, dilation=4)
        self.conv2 = DepthwiseSeparableConv(128, 256, kernel_size=3, padding=1)
        self.sn2 = SwitchNorm2d(256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(dim=2)
        x = F.relu(self.sn1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = F.relu(self.sn2(self.conv2(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return x



if __name__ == '__main__':
    # Initialize the model
    in_channels = 224
    num_classes = 10
    embedding_dim = 128
    batch_size = 20
    model = SqueezeExcitation(in_channels, embedding_dim, num_classes)
    sample_input = torch.randn(batch_size, in_channels, 160, 160)
    output = model(sample_input)
    print(output.shape)