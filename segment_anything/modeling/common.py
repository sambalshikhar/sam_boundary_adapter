
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type


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