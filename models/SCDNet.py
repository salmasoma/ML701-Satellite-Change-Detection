import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.misc import initialize_weights  # Ensure this utility script is available

# Use the previously defined functions and classes (conv1x1, conv3x3, ResBlock, etc.) from the provided example...

class MACModule(nn.Module):
    # Multi-scale Atrous Convolution Module
    def __init__(self, in_channels):
        super(MACModule, self).__init__()
        # Define the atrous convolution layers with different dilation rates
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3)

    def forward(self, x):
        # Apply the atrous convolutions and sum their outputs
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return x1 + x2 + x3

class AttentionModule(nn.Module):
    # Attention Module for feature recalibration
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        # Implement attention mechanism
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SCDNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(SCDNet, self).__init__()
        # Pre-trained ResNet34 as the backbone for the encoder
        self.backbone = models.resnet34(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Use the ResNet34 layers as part of the encoder
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        # MAC module
        self.mac_individual = MACModule(512)  

        # MAC module for concatenated feature maps
        self.mac_combined = MACModule(1024)  

        # Attention module
        self.attention = AttentionModule(512)  # Modify as per the output of your network

        # Final layers
        self.dropout = nn.Dropout()
        self.classifier = nn.Conv2d(512, 7, kernel_size=1)

        # Auxiliary classifiers for deep supervision
        self.aux1 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.aux2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.aux3 = nn.Conv2d(512, num_classes, kernel_size=1)

    
    def forward(self, x1, x2):
        x_size = x1.size()        
        # Forward pass through the SCDNet architecture
        # Pass through shared backbone layers
        x1 = self.backbone.conv1(x1)
        x2 = self.backbone.conv1(x2)

        x1 = self.layer1(x1)
        x2 = self.layer1(x2)

        x1 = self.layer2(x1)
        x2 = self.layer2(x2)

        x1 = self.layer3(x1)
        x2 = self.layer3(x2)

        x1 = self.layer4(x1)
        x2 = self.layer4(x2)

        # Apply the MAC module
        x1 = self.mac_individual(x1)
        x2 = self.mac_individual(x2)

        # Apply the attention module
        x1 = self.attention(x1)
        x2 = self.attention(x2)

        # Apply dropout
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        x = torch.cat([x1, x2], 1)  # Concatenating along the channel dimension
        change = self.mac_combined(x)  # Ensure the MAC module can handle the increased channel count
        change = self.attention(change)

        # Apply change detection classifier
        change = self.change_classifier(change)

        # Apply classification to x1 and x2
        out1 = self.classifier(x1)
        out2 = self.classifier(x2)
    
        
        return F.interpolate(change, x_size[2:], mode='bilinear'), F.interpolate(out1, x_size[2:], mode='bilinear'), F.interpolate(out2, x_size[2:], mode='bilinear')

