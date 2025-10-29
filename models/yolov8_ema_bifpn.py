import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, Bottleneck, C2f, SPPF
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import initialize_weights

from .modules.ema_attention import EMABlock
from .modules.bifpn import BiFPN


class EMAConv(nn.Module):
    """集成EMA注意力的卷积块"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, reduction=32):
        super(EMAConv, self).__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, groups=groups)
        self.ema = EMABlock(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = self.ema(x)
        return x


class EMA_C2f(nn.Module):
    """集成EMA注意力的C2f模块"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=False, groups=1, expansion=0.5, reduction=32):
        super(EMA_C2f, self).__init__()
        # 基于YOLOv8的C2f结构，在关键位置添加EMA注意力
        self.cv1 = Conv(in_channels, int(out_channels * expansion), 1, 1)
        self.cv2 = Conv((1 + n) * int(out_channels * expansion), out_channels, 1)

        # 在bottleneck中添加EMA
        self.m = nn.ModuleList([
            EMA_Bottleneck(int(out_channels * expansion), int(out_channels * expansion),
                           shortcut, groups, reduction) for _ in range(n)
        ])

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class EMA_Bottleneck(nn.Module):
    """集成EMA注意力的Bottleneck"""

    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, reduction=32):
        super(EMA_Bottleneck, self).__init__()
        self.cv1 = Conv(in_channels, out_channels, 1, 1)
        self.cv2 = Conv(out_channels, out_channels, 3, 1, groups=groups)
        self.ema = EMABlock(out_channels, reduction)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.ema(self.cv2(self.cv1(x))) if self.add else self.ema(self.cv2(self.cv1(x)))


class YOLOv8_EMA_BiFPN(DetectionModel):
    """集成EMA注意力和BiFPN的YOLOv8模型"""

    def __init__(self, cfg='yolov8s.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)

        # 在backbone的关键位置添加EMA注意力
        self._add_ema_to_backbone()

        # 用BiFPN替换原有的PANet
        self._replace_neck_with_bifpn()

        # 初始化权重
        initialize_weights(self)

    def _add_ema_to_backbone(self):
        """在backbone中添加EMA注意力"""
        # 找到backbone的模块并替换
        for name, module in self.model.named_children():
            if hasattr(module, 'm') and isinstance(module.m, nn.ModuleList):
                # 在C2f模块中添加EMA
                for i, layer in enumerate(module.m):
                    if isinstance(layer, Bottleneck):
                        # 替换为EMA_Bottleneck
                        in_channels = layer.cv1.conv.in_channels
                        out_channels = layer.cv1.conv.out_channels
                        shortcut = hasattr(layer, 'add') and layer.add

                        new_layer = EMA_Bottleneck(in_channels, out_channels, shortcut)
                        module.m[i] = new_layer

    def _replace_neck_with_bifpn(self):
        """用BiFPN替换neck部分"""
        # 获取多尺度特征通道数
        feature_channels = []
        for module in self.model:
            if isinstance(module, (C2f, SPPF)):
                feature_channels.append(module.cv2.conv.out_channels)

        # 只取最后三个尺度的特征（P3, P4, P5）
        feature_channels = feature_channels[-3:]

        # 创建BiFPN
        self.bifpn = BiFPN(feature_channels=feature_channels,
                           bifpn_channels=256,
                           num_layers=3)

    def forward(self, x, *args, **kwargs):
        # 获取多尺度特征
        features = []

        for i, module in enumerate(self.model):
            x = module(x)
            # 收集特定层的特征（对应P3, P4, P5）
            if i in [len(self.model) - 4, len(self.model) - 3, len(self.model) - 2]:
                features.append(x)

        # 通过BiFPN处理特征
        if hasattr(self, 'bifpn'):
            features = self.bifpn(features)

        # 将处理后的特征传递到head
        if hasattr(self, 'detect'):
            return self.detect(features[-1])  # 简化处理，实际应该处理所有尺度

        return x


def create_yolov8_ema_bifpn_model(pretrained=True):
    """创建集成EMA和BiFPN的YOLOv8模型"""

    # 基础配置
    cfg = {
        'nc': 10,  # VisDrone的类别数
        'depth_multiple': 0.33,  # yolov8s
        'width_multiple': 0.50,  # yolov8s
        'backbone': [
            [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
            [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
            [-1, 3, 'C2f', [128, True]],
            [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
            [-1, 6, 'C2f', [256, True]],
            [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
            [-1, 6, 'C2f', [512, True]],
            [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
            [-1, 3, 'C2f', [1024, True]],
            [-1, 1, 'SPPF', [1024, 5]],  # 9
        ],
        'head': [
            [-1, 1, 'Conv', [512, 1, 1]],
            [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
            [-1, 3, 'C2f', [512, False]],  # 13

            [-1, 1, 'Conv', [256, 1, 1]],
            [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
            [-1, 3, 'C2f', [256, False]],  # 17 (P3/8-small)

            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 14], 1, 'Concat', [1]],  # cat head P4
            [-1, 3, 'C2f', [512, False]],  # 20 (P4/16-medium)

            [-1, 1, 'Conv', [512, 3, 2]],
            [[-1, 10], 1, 'Concat', [1]],  # cat head P5
            [-1, 3, 'C2f', [1024, False]],  # 23 (P5/32-large)

            [[17, 20, 23], 1, 'Detect', ['nc']],  # Detect(P3, P4, P5)
        ]
    }

    model = YOLOv8_EMA_BiFPN(cfg)

    if pretrained:
        # 加载预训练权重（需要适配）
        try:
            from ultralytics import YOLO
            pretrained_model = YOLO('yolov8s.pt')
            # 这里需要实现权重加载逻辑
            print("加载预训练权重...")
        except:
            print("警告：无法加载预训练权重，使用随机初始化")

    return model