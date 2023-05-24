import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['ResNeXt', 'resnext50', 'resnext101', 'resnext152']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.ln1 = nn.LayerNorm(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.ln2 = nn.LayerNorm(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.ln3 = nn.LayerNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """The permute function is used to change the order of dimensions, 
        because LayerNorm by default applies normalization over the last dimension. 
        We want to apply normalization over the channels dimension, so we need to 
        move the channels dimension to the end before applying LayerNorm, 
        and then move it back to its original position afterwards.
        """
        residual = x

        out = self.conv1(x)
        out = self.ln1(out.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.ln2(out.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.ln3(out.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def resnext50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 height=160,
                 width=160,
                 shortcut_type='B',
                 cardinality=32,
                 out_features=400,
                 in_channels=224):
        super(ResNeXt, self).__init__()

        self.inplanes = in_channels
        self.conv1 = nn.Conv3d(
            in_channels,  # Change the input channels to 224
            out_channels=in_channels,
            kernel_size=(1, 7, 7),  # Update the kernel size
            stride=(1, 1, 1),
            padding=(0, 3, 3),
            bias=False)
        
        self.ln1 = nn.LayerNorm(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        # last_duration = int(math.ceil(width / 16))
        # last_size = int(math.ceil(height / 32))

        self.avgpool = nn.AvgPool3d((1, 10, 10), stride=1)
        # self.projector = nn.Linear(51200, out_features) # cardinality * 32 * block.expansion

        self.projector = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.65),
            nn.Linear(in_features=512, out_features=out_features),
        )

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                pbe = planes * block.expansion
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        pbe,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.LayerNorm(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = self.avgpool(x)

        embeddings = x.view(x.size(0), -1)  # Return embeddings directly
        return self.projector(embeddings)
