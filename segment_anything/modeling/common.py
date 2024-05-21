
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type
import math

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
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

class Adapter_inception_conv(nn.Module):
    def __init__(self, in_channels):
        super(Adapter_inception_conv, self).__init__()

        int_dim=in_channels//3
        self.conv_low_pass_1 = nn.Conv2d(in_channels,int_dim, kernel_size=(1, 1))
        self.layernorm_1=LayerNorm2d(int_dim)
        self.conv_low_pass_2 = nn.Conv2d(in_channels,int_dim, kernel_size=(1, 1))
        self.layernorm_2=LayerNorm2d(int_dim)
        self.conv_low_pass_3 = nn.Conv2d(in_channels,in_channels, kernel_size=(1, 1))
        self.layernorm_3=LayerNorm2d(in_channels)
        
        # 1x1 Convolution
        self.conv1x1 = nn.Conv2d(int_dim,int_dim, kernel_size=(1, 1))
        self.layernorm_1x1=LayerNorm2d(int_dim)

        # 3x3 Convolution
        self.conv3x3 = nn.Conv2d(int_dim,int_dim, kernel_size=(3, 3), padding=1)
        self.layernorm_3x3=LayerNorm2d(int_dim)

        # 5x5 Convolution
        self.conv5x5 = nn.Conv2d(int_dim,int_dim, kernel_size=(5, 5), padding=2)
        self.layernorm_5x5=LayerNorm2d(int_dim)
        
        
        self.proj=nn.Conv2d(in_channels,in_channels,kernel_size=(1, 1))
        self.layernorm_proj=LayerNorm2d(in_channels)
        
    def forward(self, x):
        
        conv_low_pass_1=F.relu(self.conv_low_pass_1(x.permute(0,3,1,2)))
        conv_low_pass_2=F.relu(self.conv_low_pass_2(x.permute(0,3,1,2)))
        conv_low_pass_3=F.relu(self.conv_low_pass_3(x.permute(0,3,1,2)))
        # Apply each convolutional layer
        out1x1 = F.relu(self.conv1x1(conv_low_pass_1))
        out3x3 = F.relu(self.conv3x3(conv_low_pass_1))
        out5x5 = F.relu(self.conv5x5(conv_low_pass_2))

        # Concatenate along the last axis (axis=1 in PyTorch)
        concat_feature=torch.cat([out1x1, out3x3, out5x5], dim=1)
        out_feature=F.relu(self.proj(concat_feature))
        #out_feature=out_feature
        out_feature=out_feature.permute(0,2,3,1)+x
        
        return out_feature
    

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
        # Ensure the features are in channel-first format for processing
        fc = fc.permute(0, 3, 1, 2)
        ft = ft.permute(0, 3, 1, 2)
        
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

        # Convert back to channel-last
        m = m.permute(0, 2, 3, 1)

        return m

class Adapter_inception(nn.Module):
    def __init__(self, in_channels):
        super(Adapter_inception, self).__init__()

        int_dim=in_channels//3
        self.conv_low_pass = nn.Conv2d(in_channels,int_dim, kernel_size=(1, 1))
        # 1x1 Convolution
        self.conv1x1 = nn.Conv2d(int_dim,int_dim, kernel_size=(1, 1))

        # 3x3 Convolution
        self.conv3x3 = nn.Conv2d(int_dim,int_dim, kernel_size=(3, 3), padding=1)

        # 5x5 Convolution
        self.conv5x5 = nn.Conv2d(int_dim,int_dim, kernel_size=(5, 5), padding=2)
        self.proj=nn.Linear(in_channels,in_channels)
        self.relu=nn.ReLU()
        #self.apply(self._init_weights)
        
    def forward(self, x):
        conv_low_pass=F.relu(self.conv_low_pass(x.permute(0,3,1,2)))
        # Apply each convolutional layer
        out1x1 = F.relu(self.conv1x1(conv_low_pass))
        out3x3 = F.relu(self.conv3x3(conv_low_pass))
        out5x5 = F.relu(self.conv5x5(conv_low_pass))

        # Concatenate along the last axis (axis=1 in PyTorch)
        concat_feature=torch.cat([out1x1, out3x3, out5x5], dim=1)
        concat_feature=concat_feature.permute(0,2,3,1)
        out_feature=self.relu(self.proj(concat_feature))+x
        
        return out_feature

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    
class Adapter_conv(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        
        self.act = act_layer()
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
        self.conv_fc1 = nn.Linear(D_features, D_hidden_features)
    
    
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x,conv_feature):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        x_conv = self.D_fc1(conv_feature)
        x_conv=self.act(x_conv)
        xs=xs+x_conv
        xs = self.D_fc2(xs)
        
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self._init_weights()
        #self.norm=nn.LayerNorm(D_features)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups = in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//2)*3, (inter_planes//2)*3, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicSepConv((inter_planes//2)*3, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, (inter_planes//2)*3, kernel_size=3, stride=stride, padding=1),
                BasicSepConv((inter_planes//2)*3, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(3*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        if in_planes == out_planes:
            self.identity = True
        else:
            self.identity = False
            self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x1,x2),1)
        out = self.ConvLinear(out)
        if self.identity:
            out = out*self.scale + x
        else:
            short = self.shortcut(x)
            out = out*self.scale + short
        out = self.relu(out)

        return out
