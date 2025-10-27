import torch
import torch.nn as nn


class EMA(nn.Module):
    """
    高效多尺度注意力模块
    参考论文: "Efficient Multi-Scale Attention Module"
    """

    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w

        # 水平池化和垂直池化
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # 连接并卷积
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # 注意力权重计算
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        # 全局注意力
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        # 权重融合
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class CA(nn.Module):
    """
    坐标注意力模块
    参考论文: "Coordinate Attention for Efficient Mobile Network Design"
    """

    def __init__(self, channel, reduction=32):
        super(CA, self).__init__()
        self.h_avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.w_avg_pool = nn.AdaptiveAvgPool2d((1, None))

        reduction_channel = max(channel // reduction, 8)
        self.conv1 = nn.Conv2d(channel, reduction_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(reduction_channel)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(reduction_channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(reduction_channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        b, c, h, w = x.size()

        # 水平全局平均池化
        x_h = self.h_avg_pool(x)  # [b, c, h, 1]
        # 垂直全局平均池化  
        x_w = self.w_avg_pool(x)  # [b, c, 1, w]

        # 连接和卷积
        x_cat = torch.cat([x_h, x_w], dim=2)  # [b, c, h+w, 1]
        x_cat = self.conv1(x_cat)
        x_cat = self.bn1(x_cat)
        x_cat = self.act(x_cat)

        # 分割回水平和垂直
        x_h, x_w = torch.split(x_cat, [h, w], dim=2)

        # 1x1卷积生成注意力权重
        attention_h = torch.sigmoid(self.conv_h(x_h))
        attention_w = torch.sigmoid(self.conv_w(x_w))

        # 应用注意力
        out = identity * attention_h * attention_w

        return out