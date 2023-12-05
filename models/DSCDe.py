import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights

import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights

class DSCDe(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(DSCDe, self).__init__()
        # Initialize the base ResNet model
        resnet = models.resnet34(pretrained=pretrained)
        # Replace the first convolutional layer of ResNet to accommodate custom in_channels
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, :min(in_channels, 3), :, :].copy_(resnet.conv1.weight.data[:, :min(in_channels, 3), :, :])
        # Initialize other layers from ResNet
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # Classifier layer to produce segmentation map
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

        initialize_weights(self.classifier)

    def forward(self, x1, x2):
        # Process input I1
        x1 = self.layer0(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        out1 = self.classifier(x1)

        # Process input I2 using the same layers
        x2 = self.layer0(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        out2 = self.classifier(x2)

        # Upsample to match the input size and return
        x_size = x1.size()  # Assuming x1 and x2 have the same size
        target_size = (512, 512)  # Replace with the actual size of your ground truth

        out1_upsampled = F.interpolate(out1, size=target_size, mode='bilinear', align_corners=True)
        out2_upsampled = F.interpolate(out2, size=target_size, mode='bilinear', align_corners=True)

        change = torch.zeros_like(out1_upsampled)

        # Assuming you have a binary change map output that needs to be upsampled as well
        change_upsampled = F.interpolate(change, size=target_size, mode='bilinear', align_corners=True)

        return change_upsampled, out1_upsampled, out2_upsampled
