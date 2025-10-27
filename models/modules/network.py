import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN_Block(nn.Module):
    """
    BiFPN (加权双向特征金字塔网络) 模块
    参考论文: "EfficientDet: Scalable and Efficient Object Detection"
    """

    def __init__(self, channels, levels=5, epsilon=1e-4):
        super(BiFPN_Block, self).__init__()
        self.epsilon = epsilon
        self.levels = levels

        # 上采样和下采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 可学习的权重（快速归一化融合）
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)  # 用于双向融合
            for _ in range(levels)
        ])

        # 卷积层用于特征融合
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(levels)
        ])

    def fast_normalized_fusion(self, features, weights, level):
        """快速归一化融合"""
        normalized_weights = F.relu(weights[level])
        weight_sum = torch.sum(normalized_weights) + self.epsilon

        # 加权融合
        fused = sum(w / weight_sum * f for w, f in zip(normalized_weights, features))
        return fused

    def forward(self, inputs):
        """
        inputs: 多尺度特征图列表 [P3, P4, P5, ...]
        """
        assert len(inputs) == self.levels

        # 自上而下路径
        top_down = []
        for i in range(self.levels - 1, -1, -1):
            if i == self.levels - 1:  # 最顶层
                top_down.append(inputs[i])
            else:
                # 上采样并融合
                up_feat = self.upsample(top_down[-1])
                fused = self.fast_normalized_fusion([inputs[i], up_feat], self.weights, i)
                top_down.append(self.conv_blocks[i](fused))

        top_down = list(reversed(top_down))

        # 自下而上路径
        bottom_up = []
        for i in range(self.levels):
            if i == 0:  # 最底层
                bottom_up.append(top_down[i])
            else:
                # 下采样并融合
                down_feat = self.downsample(bottom_up[-1])
                fused = self.fast_normalized_fusion([top_down[i], down_feat], self.weights, i)
                bottom_up.append(self.conv_blocks[i](fused))

        return bottom_up


class SPPF_Enhanced(nn.Module):
    """
    增强型空间金字塔池化模块
    在YOLOv8 SPPF基础上增加多尺度特征增强
    """

    def __init__(self, channels, pool_sizes=(5, 9, 13)):
        super(SPPF_Enhanced, self).__init__()

        self.pool_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2)
            for size in pool_sizes
        ])

        self.conv_reduce = nn.Conv2d(channels * (len(pool_sizes) + 1), channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        features = [x]
        for pool in self.pool_layers:
            features.append(pool(x))

        # 通道维度拼接
        out = torch.cat(features, dim=1)
        out = self.conv_reduce(out)
        out = self.bn(out)
        out = self.act(out)

        return out