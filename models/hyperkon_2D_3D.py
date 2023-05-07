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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        x = F.relu(x)
        return x

class HyperKon_2D_3D(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super(HyperKon_2D_3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.res_block1 = ResidualBlock2D(64, 128)
        self.res_block2 = ResidualBlock2D(128, 256)
        self.conv2 = nn.Conv3d(256, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.bn2 = nn.BatchNorm3d(512)
        self.gap = nn.AdaptiveAvgPool3d(1)
        # self.projector = nn.Linear(512, embedding_dim)
        # self.softmax = nn.Linear(embedding_dim, num_classes)

        self.projector = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.65),
            nn.Linear(in_features=512, out_features=embedding_dim),
        )

    def forward(self, x):
        # Process spatial information
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.squeeze(dim=2)
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Process spectral information
        x = x.unsqueeze(dim=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projector(x)
        # x = self.softmax(x)
        return x

if __name__ == "__main__":
    # Initialize the model
    in_channels = 224
    num_classes = 10
    embedding_dim = 128
    model = HyperKon_2D_3D(in_channels, embedding_dim)

    # Print the model
    print(model)
