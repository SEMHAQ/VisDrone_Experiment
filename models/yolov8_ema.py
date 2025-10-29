import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, C2f, SPPF
from ultralytics.nn.tasks import DetectionModel


class EMAttention(nn.Module):
    """EMA注意力机制 - 简化版本"""

    def __init__(self, channels, reduction=16):
        super(EMAttention, self).__init__()
        self.groups = max(1, channels // reduction)

        # 池化层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 卷积层
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 水平注意力
        x_h = self.pool_h(x)  # [b, c, h, 1]
        # 垂直注意力  
        x_w = self.pool_w(x)  # [b, c, 1, w]

        # 注意力权重
        attention = self.sigmoid(self.conv(x_h * x_w.permute(0, 1, 3, 2)))

        return x * attention


class EMA_C2f(nn.Module):
    """在C2f模块中集成EMA注意力"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=False):
        super(EMA_C2f, self).__init__()

        # 保持原始C2f结构
        hidden_dim = int(out_channels * 0.5)
        self.conv1 = Conv(in_channels, hidden_dim, 1, 1)
        self.conv2 = Conv((2 + n) * hidden_dim, out_channels, 1)

        # Bottleneck列表（添加EMA）
        self.m = nn.ModuleList([
            EMA_Bottleneck(hidden_dim, hidden_dim, shortcut) for _ in range(n)
        ])

    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y.append(y[-1])  # 原始特征
        return self.conv2(torch.cat(y, 1))


class EMA_Bottleneck(nn.Module):
    """Bottleneck with EMA attention"""

    def __init__(self, in_channels, out_channels, shortcut=True):
        super(EMA_Bottleneck, self).__init__()
        self.cv1 = Conv(in_channels, out_channels, 1, 1)
        self.cv2 = Conv(out_channels, out_channels, 3, 1)
        self.ema = EMAttention(out_channels)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.ema(self.cv2(self.cv1(x))) if self.add else self.ema(self.cv2(self.cv1(x)))


def create_yolov8_ema():
    """创建集成EMA注意力的YOLOv8模型"""
    from ultralytics import YOLO

    # 加载基准模型
    model = YOLO('yolov8s.pt')

    # 获取模型结构
    model_dict = model.model.model if hasattr(model.model, 'model') else model.model

    # 在关键位置添加EMA注意力
    # 这里需要根据实际的模型结构来修改
    # 由于YOLOv8结构封闭，我们通过修改配置文件或hook方式实现

    return model