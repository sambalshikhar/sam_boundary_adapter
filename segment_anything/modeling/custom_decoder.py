import torch
import torch.nn as nn
import torch.nn.functional as F

class upDecoder(nn.Module):
    def __init__(self):
        super(upDecoder, self).__init__()
        # Define the decoder layers
        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # Output size: (N, 512, 128, 128)
            nn.Conv2d(256,128, kernel_size=3, padding=1),           # Output size: (N, 512, 128, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.up_2=nn.Sequential(
            nn.ConvTranspose2d(128,128, kernel_size=2, stride=2),  # Output size: (N, 256, 256, 256)
            nn.Conv2d(128,64, kernel_size=3, padding=1),           # Output size: (N, 256, 256, 256)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.conv_1=nn.Sequential(
            nn.Conv2d(64,32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv_2=nn.Sequential(
            nn.Conv2d(32,16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,1, kernel_size=1)                          # Output size: (N, 1, 1024, 1024)
        )
        self.merge_1=nn.Sequential(
            nn.Conv2d(256*2,256, kernel_size=1),
            nn.ReLU()
        )
        self.merge_2=nn.Sequential(
            nn.Conv2d(128*2,128, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x,conv_1,conv_2):
        x=F.normalize(torch.concat([x,conv_1],axis=1),1)
        x=self.merge_1(x)
        x=self.up_1(x)
        x=F.normalize(torch.concat([x,conv_2],axis=1),1)
        x=self.merge_2(x)
        x=self.up_2(x)
        x=self.conv_1(x)
        x=self.conv_2(x)
        return x
    
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