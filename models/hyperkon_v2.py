import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        # Channel attention module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial attention module
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = avg_out + max_out
        x = x * channel_out

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.conv(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_out
        return x



class ModifiedResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ModifiedResidualBlock2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.depthwise2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.pointwise2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.cbam = CBAM(out_channels)
        
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.pointwise(self.depthwise(x))))
        x = self.bn2(self.pointwise2(self.depthwise2(x)))
        x = self.se(x)
        x = self.cbam(x)
        x += self.shortcut(residual)
        x = F.relu(x)
        return x


class HyperKon_2D_3D(nn.Module):
    def __init__(self, in_channels, out_features):
        super(HyperKon_2D_3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.res_block1 = ModifiedResidualBlock2D(64, 128)
        self.res_block2 = ModifiedResidualBlock2D(128, 256)
        self.res_block3 = ModifiedResidualBlock2D(256, 256)
        self.res_block4 = ModifiedResidualBlock2D(256, 512)
        self.conv2 = nn.Conv3d(512, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.bn2 = nn.BatchNorm3d(512)
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.projector = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.65),
            nn.Linear(in_features=512, out_features=out_features),
        )

    def forward(self, x):
        x = x.unsqueeze(dim=2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.squeeze(dim=2)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        x = x.unsqueeze(dim=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projector(x)
        return x

    
if __name__ == '__main__':
    in_channels = 1
    in_channels2 = 224
    num_classes = 128
    embedding_dim = 256
    batch_size = 20
    m1 = HyperKon_2D_3D(in_channels, num_classes)
    m2 = HyperKon_2D_3D(in_channels2, num_classes)

    sample_input = torch.randn(batch_size, in_channels, 32, 32)
    sample_input2 = torch.randn(batch_size, in_channels2, 32, 32)
    output = m1(sample_input)
    output2 = m2(sample_input2)
    print(output.shape)
    print(output2.shape)
    