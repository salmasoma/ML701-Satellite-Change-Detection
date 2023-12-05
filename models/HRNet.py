import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models import build_backbone
from utils.misc import initialize_weights

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class HRNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(HRNet, self).__init__()
        # Initialize HRNet

        # Configuration for HRNet
        hrnet_cfg = {
            'type': 'HRNet',
            'extra': {
                'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': [4], 'num_channels': [64]},
                'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': [4, 4], 'num_channels': [18, 36]},
                'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': [4, 4, 4], 'num_channels': [18, 36, 72]},
                'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': [4, 4, 4, 4], 'num_channels': [18, 36, 72, 144]}
            },
            'pretrained': None  # Set to the path of the pretrained weights if available
        }

        # Initialize HRNet
        self.HRNet = build_backbone(hrnet_cfg)

        hrnet_out_channels = 144

        self.classifier1 = nn.Conv2d(hrnet_out_channels, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(hrnet_out_channels, num_classes, kernel_size=1)
        self.resCD = self._make_layer(ResBlock, 288, 144, 6, stride=1) # first ResBlock should expect 288 channels
        self.classifierCD = nn.Sequential(
            nn.Conv2d(hrnet_out_channels, hrnet_out_channels // 2, kernel_size=1),
            nn.BatchNorm2d(hrnet_out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(hrnet_out_channels // 2, 1, kernel_size=1)
        )

        initialize_weights(self.classifier1, self.classifier2, self.resCD, self.classifierCD)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def base_forward(self, x):
        # Forward pass through HRNet
        x = self.HRNet.forward(x)
        return x

    def CD_forward(self, x1, x2):
        # Concatenating features from two HRNet models
        x = torch.cat([x1, x2], dim=1)  # Concatenate along the channel dimension

        # Pass through the resCD layers
        x = self.resCD(x)


        change = self.classifierCD(x)
        return change


    def forward(self, x1, x2):
        assert x1.dim() == 4 and x2.dim() == 4, "Input tensors must be 4D"

        # Process through HRNet
        x1_features = self.HRNet(x1)
        x2_features = self.HRNet(x2)

        # Assuming you want to use the last feature map from HRNet
        x1_last_feature = x1_features[-1]
        x2_last_feature = x2_features[-1]

        # Debug: Check the dimensionality of the last features
        assert x1_last_feature.dim() == 4 and x2_last_feature.dim() == 4, "HRNet outputs must be 4D"

        change = self.CD_forward(x1_last_feature, x2_last_feature)

        # Debug: Check the dimensionality after CD_forward
        assert change.dim() == 4, "Change map must be 4D"

        out1 = self.classifier1(x1_last_feature)
        out2 = self.classifier2(x2_last_feature)

        return (F.upsample(change, x1.size()[2:], mode='bilinear'), 
                F.upsample(out1, x1.size()[2:], mode='bilinear'), 
                F.upsample(out2, x1.size()[2:], mode='bilinear'))
