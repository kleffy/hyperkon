import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else None

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.se:
            x = self.se(x)
        x += self.shortcut(residual)
        x = F.relu(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, embedding_dim=512, num_classes=10):
        super(SqueezeExcitation, self).__init__()
        self.conv1 = DepthwiseSeparableConv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res_block1 = ResidualBlock(64, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.conv2 = DepthwiseSeparableConv2d(256, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(dim=2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return x

if __name__ == '__main__':
    # Example usage
    in_channels = 1
    in_channels2 = 224
    num_classes = 128
    embedding_dim = 256
    batch_size = 20
    m1 = SqueezeExcitation(in_channels, embedding_dim, num_classes)
    m2 = SqueezeExcitation(in_channels2, embedding_dim, num_classes)

    # Test the model with a sample input
    sample_input = torch.randn(batch_size, in_channels, 32, 32)
    sample_input2 = torch.randn(batch_size, in_channels2, 32, 32)
    output = m1(sample_input)
    output2 = m2(sample_input2)
    print(output.shape)
    print(output2.shape)
