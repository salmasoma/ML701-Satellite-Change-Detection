import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.misc import initialize_weights

# Helper Functions
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# FCN Module
class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        # Initialize ResNet with pretrained weights
        resnet = models.resnet34(pretrained=pretrained)
        
        # Replace first conv layer if in_channels is not 3
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Using parts of ResNet as layers
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Additional head layer
        self.head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        initialize_weights(self)

    def base_forward(self, x):
        x = self.layer0(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x

# ResBlock Module
class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
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

# DualStreamFCN Model
class DualStreamFCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(DualStreamFCN, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, num_classes, kernel_size=1)
        
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.classifierCD = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Conv2d(64, 1, kernel_size=1)
        )

        initialize_weights(self)

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

    def CD_forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.resCD(x)
        change = self.classifierCD(x)
        return change

    def forward(self, x1, x2):
        x_size = x1.size()
        
        x1 = self.FCN.base_forward(x1)
        x2 = self.FCN.base_forward(x2)

        change = self.CD_forward(x1, x2)
                
        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)
        
        return (F.upsample(change, x_size[2:], mode='bilinear'), 
                F.upsample(out1, x_size[2:], mode='bilinear'), 
                F.upsample(out2, x_size[2:], mode='bilinear'))
