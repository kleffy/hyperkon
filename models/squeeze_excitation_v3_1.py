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
    def __init__(self, in_channels, out_channels, use_bam=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.4)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bam = BAMBlock(out_channels) if use_bam else None

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
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        
        if self.bam:
            x = self.bam(x)
        x += self.shortcut(residual)
        x = F.relu(x)
        return x

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, out_features=10):
        super(SqueezeExcitation, self).__init__()
        print(f'initializing SqueezeExcitation ...')
        # self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv1 = DepthwiseSeparableConv2d(in_channels, 32, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)
        # self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv2 = DepthwiseSeparableConv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        # x = x.squeeze(dim=2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return x

if __name__ == '__main__':
    from torchinfo import summary
    in_channels = 224
    out_features = 128
    batch_size = 10
    height, width = 32, 32
    model = SqueezeExcitation(in_channels=in_channels, out_features=out_features)
    dummy_input = torch.randn(batch_size, in_channels, height, width)
    summary(model, input_size=(1,in_channels, height, width),col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=10,row_settings=['var_names'],depth=4)
    

    # Test the model with a sample input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = model(dummy_input.to(device))
    print(output.shape)