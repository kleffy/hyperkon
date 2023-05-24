import torch
import torch.nn as nn
import torch.nn.functional as F

class BAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
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
    def __init__(self, in_channels, out_channels, use_bam=True, kernel_size=3, dropout_p=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm3 = nn.InstanceNorm2d(out_channels)
        self.bam = BAMBlock(out_channels) if use_bam else None
        self.dropout = nn.Dropout(dropout_p)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        if self.bam:
            x = self.bam(x)
        x += self.shortcut(residual)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, num_classes=10, dropout_p=0.5):
        super(SqueezeExcitation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(64)
        self.res_block1 = ResidualBlock(64, 128, kernel_size=3, dropout_p=dropout_p)
        self.res_block2 = ResidualBlock(128, 256, kernel_size=3, dropout_p=dropout_p)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(dim=2)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # x = x.unsqueeze(dim=2)
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return x


if __name__ == '__main__':
    # Initialize the model
    in_channels = 224
    num_classes = 10
    embedding_dim = 256
    batch_size = 128
    model = SqueezeExcitation(in_channels, embedding_dim, num_classes)
    sample_input = torch.randn(batch_size, in_channels, 160, 160)
    output = model(sample_input)
    print(output.shape)