import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationDecoder(nn.Module):
    def __init__(self, vit_embedding_size=256, cnn_embedding_size=256):
        super(SegmentationDecoder, self).__init__()

        # Upsampling blocks
        self.upconv1 = nn.ConvTranspose2d(vit_embedding_size, 128, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        #self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.mlp=nn.Linear(vit_embedding_size*2,vit_embedding_size)
       
        # Final convolution for segmentation map
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, vit_embedding,cnn_embedding):
        #print(vit_embedding.size(),cnn_embedding.size())
        # Upsampling with concatenation
        x = torch.cat((vit_embedding, cnn_embedding), dim=1)
        x = self.mlp(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = F.relu(x)

        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        # Final convolution for segmentation map
        segmentation_map = self.final_conv(x)

        return segmentation_map
