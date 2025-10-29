import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN_Module(nn.Module):
    """BiFPN模块 - 替换YOLOv8的PANet"""

    def __init__(self, feature_channels=[256, 512, 1024], bifpn_channels=256):
        super(BiFPN_Module, self).__init__()

        self.P3_channels = feature_channels[0]
        self.P4_channels = feature_channels[1]
        self.P5_channels = feature_channels[2]
        self.bifpn_channels = bifpn_channels

        # 输入投影
        self.p3_proj = nn.Conv2d(self.P3_channels, bifpn_channels, 1)
        self.p4_proj = nn.Conv2d(self.P4_channels, bifpn_channels, 1)
        self.p5_proj = nn.Conv2d(self.P5_channels, bifpn_channels, 1)

        # 自上而下路径
        self.p4_td_conv = nn.Sequential(
            nn.Conv2d(bifpn_channels, bifpn_channels, 3, padding=1),
            nn.BatchNorm2d(bifpn_channels),
            nn.SiLU(inplace=True)
        )
        self.p3_td_conv = nn.Sequential(
            nn.Conv2d(bifpn_channels, bifpn_channels, 3, padding=1),
            nn.BatchNorm2d(bifpn_channels),
            nn.SiLU(inplace=True)
        )

        # 自下而上路径
        self.p4_bu_conv = nn.Sequential(
            nn.Conv2d(bifpn_channels, bifpn_channels, 3, padding=1),
            nn.BatchNorm2d(bifpn_channels),
            nn.SiLU(inplace=True)
        )
        self.p5_bu_conv = nn.Sequential(
            nn.Conv2d(bifpn_channels, bifpn_channels, 3, padding=1),
            nn.BatchNorm2d(bifpn_channels),
            nn.SiLU(inplace=True)
        )

        # 输出投影
        self.p3_out = nn.Conv2d(bifpn_channels, self.P3_channels, 1)
        self.p4_out = nn.Conv2d(bifpn_channels, self.P4_channels, 1)
        self.p5_out = nn.Conv2d(bifpn_channels, self.P5_channels, 1)

        # 权重参数
        self.p4_td_weight = nn.Parameter(torch.ones(2))
        self.p3_td_weight = nn.Parameter(torch.ones(2))
        self.p4_bu_weight = nn.Parameter(torch.ones(2))
        self.p5_bu_weight = nn.Parameter(torch.ones(2))

        self.epsilon = 1e-4

    def weighted_fusion(self, features, weights):
        """加权特征融合"""
        normalized_weights = F.relu(weights)
        weight_sum = torch.sum(normalized_weights) + self.epsilon
        return sum(w / weight_sum * f for w, f in zip(normalized_weights, features))

    def forward(self, inputs):
        """
        inputs: [P3, P4, P5] 多尺度特征
        """
        # 输入投影
        p3_in = self.p3_proj(inputs[0])
        p4_in = self.p4_proj(inputs[1])
        p5_in = self.p5_proj(inputs[2])

        # 自上而下路径
        p5_td = p5_in
        p4_td = self.p4_td_conv(self.weighted_fusion(
            [p4_in, F.interpolate(p5_td, scale_factor=2, mode='nearest')],
            self.p4_td_weight
        ))
        p3_td = self.p3_td_conv(self.weighted_fusion(
            [p3_in, F.interpolate(p4_td, scale_factor=2, mode='nearest')],
            self.p3_td_weight
        ))

        # 自下而上路径
        p3_out = p3_td
        p4_out = self.p4_bu_conv(self.weighted_fusion(
            [p4_td, F.interpolate(p3_out, scale_factor=0.5, mode='nearest')],
            self.p4_bu_weight
        ))
        p5_out = self.p5_bu_conv(self.weighted_fusion(
            [p5_td, F.interpolate(p4_out, scale_factor=0.5, mode='nearest')],
            self.p5_bu_weight
        ))

        # 输出投影
        p3_final = self.p3_out(p3_out) + inputs[0]  # 残差连接
        p4_final = self.p4_out(p4_out) + inputs[1]
        p5_final = self.p5_out(p5_out) + inputs[2]

        return [p3_final, p4_final, p5_final]