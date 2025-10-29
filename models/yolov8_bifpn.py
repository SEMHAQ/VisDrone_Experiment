import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN_Module(nn.Module):
    """BiFPN模块 - 替换YOLOv8的PANet"""

    def __init__(self, channels_list=[256, 512, 1024], bifpn_channels=256):
        super(BiFPN_Module, self).__init__()

        self.P3_channels = channels_list[0]
        self.P4_channels = channels_list[1]
        self.P5_channels = channels_list[2]

        # 输入投影
        self.p3_proj = nn.Conv2d(self.P3_channels, bifpn_channels, 1)
        self.p4_proj = nn.Conv2d(self.P4_channels, bifpn_channels, 1)
        self.p5_proj = nn.Conv2d(self.P5_channels, bifpn_channels, 1)

        # 自上而下路径
        self.p4_td_conv = nn.Conv2d(bifpn_channels, bifpn_channels, 3, padding=1)
        self.p3_td_conv = nn.Conv2d(bifpn_channels, bifpn_channels, 3, padding=1)

        # 自下而上路径
        self.p4_bu_conv = nn.Conv2d(bifpn_channels, bifpn_channels, 3, padding=1)
        self.p5_bu_conv = nn.Conv2d(bifpn_channels, bifpn_channels, 3, padding=1)

        # 输出投影
        self.p3_out = nn.Conv2d(bifpn_channels, self.P3_channels, 1)
        self.p4_out = nn.Conv2d(bifpn_channels, self.P4_channels, 1)
        self.p5_out = nn.Conv2d(bifpn_channels, self.P5_channels, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        # inputs: [P3, P4, P5]
        p3, p4, p5 = inputs

        # 输入投影
        p3_in = self.p3_proj(p3)
        p4_in = self.p4_proj(p4)
        p5_in = self.p5_proj(p5)

        # 自上而下路径
        p5_td = p5_in
        p4_td = self.p4_td_conv(p4_in + self.upsample(p5_td))
        p3_td = self.p3_td_conv(p3_in + self.upsample(p4_td))

        # 自下而上路径
        p3_out = p3_td
        p4_out = self.p4_bu_conv(p4_td + self.downsample(p3_out))
        p5_out = self.p5_bu_conv(p5_td + self.downsample(p4_out))

        # 输出投影
        p3_final = self.p3_out(p3_out)
        p4_final = self.p4_out(p4_out)
        p5_final = self.p5_out(p5_out)

        return [p3_final, p4_final, p5_final]


def create_yolov8_bifpn():
    """创建集成BiFPN的YOLOv8模型"""
    from ultralytics import YOLO

    model = YOLO('yolov8s.pt')

    # 这里需要通过hook或修改模型结构来替换PANet为BiFPN
    # 由于YOLOv8结构封闭，实际实现较复杂

    return model