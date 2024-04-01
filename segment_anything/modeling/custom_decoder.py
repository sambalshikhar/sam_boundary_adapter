import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=1):
        super(DecoderBlock, self).__init__()

        # First upsampling layer using bilinear interpolation
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Convolutional layer between the upsampling layers
        self.conv_between = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.relu_between = nn.ReLU()

        # Second upsampling layer using bilinear interpolation
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels // 2, in_channels//4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels//4, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #x=x.permute(0,3,1,2)
        x = self.upsample1(x)
        x = self.relu_between(self.conv_between(x))
        x = self.upsample2(x)
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        return x