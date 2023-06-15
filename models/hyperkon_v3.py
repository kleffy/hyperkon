import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvBlockAttentionModule(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ConvBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv3D, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SpectralSpatialTransformerNetwork(nn.Module):
    def __init__(self, in_channels, nhead, num_layers):
        super(SpectralSpatialTransformerNetwork, self).__init__()
        self.transformer = TransformerEncoder(TransformerEncoderLayer(d_model=in_channels, nhead=nhead), num_layers=num_layers)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)  # reshape and transpose for transformer
        x = self.transformer(x)
        x = x.permute(1, 2, 0).view(x.size(1), x.size(2), int(x.size(0)**0.5), int(x.size(0)**0.5))  # transpose and reshape back
        return x

class HyperKon_V3(nn.Module):
    def __init__(self, in_channels, transformer_nhead, transformer_num_layers, cbam_channels, conv3d_out_channels):
        super(HyperKon_V3, self).__init__()
        # self.sstn = SpectralSpatialTransformerNetwork(in_channels, transformer_nhead, transformer_num_layers)
        self.cbam = ConvBlockAttentionModule(cbam_channels)
        self.conv3d = DepthwiseSeparableConv3D(cbam_channels, conv3d_out_channels)
        self.fc = nn.Linear(6553600, conv3d_out_channels)

    def forward(self, x):
        # x = self.sstn(x)
        x = self.cbam(x.squeeze(2))  # add an extra dimension for 3D convolution
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x
