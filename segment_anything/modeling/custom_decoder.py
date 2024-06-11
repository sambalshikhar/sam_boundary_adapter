import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings


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

        self.up_3=nn.Sequential(
            nn.ConvTranspose2d(64,64, kernel_size=2, stride=2),  # Output size: (N, 256, 256, 256)
            nn.Conv2d(64,32, kernel_size=3, padding=1),           # Output size: (N, 256, 256, 256)
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.conv_1=nn.Sequential(
            nn.Conv2d(32,16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,1, kernel_size=1) 
        )
        self.conv_2=nn.Sequential(
            nn.Conv2d(16,8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8,1, kernel_size=1)                          # Output size: (N, 1, 1024, 1024)
        )
        self.merge_1=nn.Sequential(
            nn.Conv2d(256*2,256, kernel_size=1),
            nn.ReLU()
        )
        self.merge_2=nn.Sequential(
            nn.Conv2d(128*2,128, kernel_size=1),
            nn.ReLU()
        )
        self.merge_3=nn.Sequential(
            nn.Conv2d(64*2,64, kernel_size=1),
            nn.ReLU()
        )
        self.apply(self._init_weights)
        
    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, x,conv_1,conv_2,conv_3):
        x=F.normalize(torch.concat([x,conv_1],axis=1),1)
        x=self.merge_1(x)
        x=self.up_1(x)
        x=F.normalize(torch.concat([x,conv_2],axis=1),1)
        x=self.merge_2(x)
        x=self.up_2(x)
        x=F.normalize(torch.concat([x,conv_3],axis=1),1)
        x=self.merge_3(x)
        x=self.up_3(x)
        x=self.conv_1(x)
        #x=self.conv_2(x)
        return x

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class SAGate(nn.Module):
    def __init__(self, channel_dim):
        super(SAGate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channel_dim * 2, channel_dim * 2),
            nn.ReLU(),
            nn.Linear(channel_dim * 2, channel_dim * 2)
        )
        self.spatial_gate_conv = nn.Conv2d(channel_dim * 2, channel_dim * 2, kernel_size=1, stride=1, groups=2)

    def forward(self, fc, ft):
        # Ensure the features are in channel-first format for processin
        
        # Feature Separation Part
        f_concat = torch.cat((fc, ft), dim=1) # Assuming the features are of the same spatial dimensions
        i = F.adaptive_avg_pool2d(f_concat, (1, 1)).view(f_concat.size(0), -1) # Global descriptor
        attention_vector = torch.sigmoid(self.mlp(i)).view(f_concat.size(0), -1, 1, 1)
        w_c, w_t = attention_vector.chunk(2, dim=1)

        filter_c = fc * w_c
        filter_t = ft * w_t

        rec_c = filter_c + fc
        rec_t = filter_t + ft

        # Feature Aggregation Part
        f_concate = torch.cat((rec_c, rec_t), dim=1) # Concatenation at specific spatial positions
        spatial_gates = self.spatial_gate_conv(f_concate)
        
        # We will now reshape to apply softmax on the correct axis
        b, c, h, w = spatial_gates.size()
        spatial_gates = spatial_gates.view(b, 2, c // 2, h, w)
        ac, at = F.softmax(spatial_gates, dim=2).chunk(2, dim=1)
        ac, at = ac.squeeze(1), at.squeeze(1)

        # Weighted sum
        m = rec_c * ac + rec_t * at
        
        return m
"""    
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

        self.up_3=nn.Sequential(
            nn.ConvTranspose2d(64,64, kernel_size=2, stride=2),  # Output size: (N, 256, 256, 256)
            nn.Conv2d(64,32, kernel_size=3, padding=1),           # Output size: (N, 256, 256, 256)
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.conv_1=nn.Sequential(
            nn.Conv2d(32,16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,1, kernel_size=1) 
        )
        self.conv_2=nn.Sequential(
            nn.Conv2d(16,8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8,1, kernel_size=1)                          # Output size: (N, 1, 1024, 1024)
        )
        self.merge_1=nn.Sequential(
            nn.Conv2d(256,256, kernel_size=1),
            nn.ReLU()
        )
        self.merge_2=nn.Sequential(
            nn.Conv2d(128,128, kernel_size=1),
            nn.ReLU()
        )
        self.merge_3=nn.Sequential(
            nn.Conv2d(64,64, kernel_size=1),
            nn.ReLU()
        )
        
        self.sa_gate_1=SAGate(channel_dim=256)
        self.sa_gate_2=SAGate(channel_dim=128)
        self.sa_gate_3=SAGate(channel_dim=64)
    def forward(self, x,conv_1,conv_2,conv_3):
        x=self.sa_gate_1(x,conv_1)
        #x=self.merge_1(x)
        x=self.up_1(x)
        #x=F.normalize(torch.concat([x,conv_2],axis=1),1)
        x=self.sa_gate_2(x,conv_2)
        #x=self.merge_2(x)
        x=self.up_2(x)
        #x=F.normalize(torch.concat([x,conv_3],axis=1),1)
        x=self.sa_gate_3(x,conv_3)
        #x=self.merge_3(x)
        x=self.up_3(x)
        x=self.conv_1(x)
        #x=self.conv_2(x)
        return x
"""

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


    
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