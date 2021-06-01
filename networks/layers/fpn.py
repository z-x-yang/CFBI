import torch
import torch.nn.functional as F
import math
from torch import nn


class FPN(nn.Module):
    def __init__(self, in_dim_4x, in_dim_8x, in_dim_16x, out_dim):
        super(FPN, self).__init__()
        self.toplayer = self._make_layer(in_dim_16x, out_dim)
        self.latlayer1 = self._make_layer(in_dim_8x, out_dim)
        self.latlayer2 = self._make_layer(in_dim_4x, out_dim)

        self.smooth1 = self._make_layer(out_dim, out_dim, kernel_size=3, padding=1)
        self.smooth2 = self._make_layer(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, x_4x, x_8x, x_16x):

        x_16x = self.toplayer(x_16x)
        x_8x = self.latlayer1(x_8x)
        x_4x = self.latlayer2(x_4x)

        x_8x = x_8x + F.interpolate(x_16x, size=x_8x.size()[-2:], mode='bilinear', align_corners=True)
        x_4x = x_4x + F.interpolate(x_8x, size=x_4x.size()[-2:], mode='bilinear', align_corners=True)

        x_8x = self.smooth1(x_8x)
        x_4x = self.smooth2(x_4x)

        return F.relu(x_4x), F.relu(x_8x), F.relu(x_16x)


    def _make_layer(self, in_dim, out_dim, kernel_size=1, padding=0):
        return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                nn.GroupNorm(32, out_dim)
            )

        

