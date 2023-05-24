import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        x = F.relu(x)
        return x

class HyperKon_2D_3D(nn.Module):
    def __init__(self, in_channels, out_features):
        super(HyperKon_2D_3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.res_block1 = ResidualBlock2D(64, 128)
        self.res_block2 = ResidualBlock2D(128, 256)
        self.res_block3 = ResidualBlock2D(256, 256)
        self.res_block4 = ResidualBlock2D(256, 512)
        self.conv2 = nn.Conv3d(512, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.bn2 = nn.BatchNorm3d(512)
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.projector = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=out_features),
        )

    def forward(self, x):
        x = x.unsqueeze(dim=2)
        # Process spatial information
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = x.squeeze(dim=2)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        # Process spectral information
        x = x.unsqueeze(dim=2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projector(x)
        return x


if __name__ == '__main__':
    # Example usage
    in_channels = 1
    in_channels2 = 224
    num_classes = 128
    embedding_dim = 256
    batch_size = 20
    m1 = HyperKon_2D_3D(in_channels, num_classes)
    m2 = HyperKon_2D_3D(in_channels2, num_classes)

    # Test the model with a sample input
    sample_input = torch.randn(batch_size, in_channels, 32, 32)
    sample_input2 = torch.randn(batch_size, in_channels2, 32, 32)
    output = m1(sample_input)
    output2 = m2(sample_input2)
    print(output.shape)
    print(output2.shape)