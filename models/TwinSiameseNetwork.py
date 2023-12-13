import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

    def forward(self, x):
        # Forward pass through the initial layers
        x = self.layer0(x)
        x = self.maxpool(x)

        # Forward pass through ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
class TwinSiameseNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(TwinSiameseNetwork, self).__init__()
        # Initialize the FCN as the backbone for feature extraction
        self.FCN = FCN(in_channels, pretrained=True)

        # Additional layers if needed for processing features before comparison
        # ...

        # Change detection layer (can be modified based on the exact requirement)
        self.change_detection = nn.Conv2d(512, 1, kernel_size=1)

        # Classifiers for each output
        self.classifier1 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward_once(self, x):
        # Forward pass for one input through the FCN backbone
        x = self.FCN(x)
        return x

    def forward(self, x1, x2):
        # Process both inputs through the same network
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)

        # Compute the 'change' map
        # This can be a simple subtraction or a more complex operation depending on the requirement
        change = torch.abs(output1 - output2)
        change = self.change_detection(change)

        # Classify each output
        out1 = self.classifier1(output1)
        out2 = self.classifier2(output2)

        # Resize outputs to match input size
        x_size = x1.size()
        return F.interpolate(change, x_size[2:], mode='bilinear'), \
               F.interpolate(out1, x_size[2:], mode='bilinear'), \
               F.interpolate(out2, x_size[2:], mode='bilinear')

    def get_similarity(self, output1, output2):
        # Calculate similarity (e.g., using cosine similarity)
        return F.cosine_similarity(output1, output2)
