import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAttention(nn.Module):
    """EMA注意力机制 - 简化版本"""

    def __init__(self, channels, reduction=16):
        super(EMAAttention, self).__init__()
        self.groups = max(1, channels // reduction)

        # 池化层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 卷积层
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 水平池化
        x_h = self.pool_h(x)  # [b, c, h, 1]
        # 垂直池化
        x_w = self.pool_w(x)  # [b, c, 1, w]

        # 计算注意力权重
        x_h_expanded = x_h.expand(-1, -1, -1, w)
        x_w_expanded = x_w.expand(-1, -1, h, -1)

        attention = self.sigmoid(self.conv(x_h_expanded * x_w_expanded))

        return x * attention