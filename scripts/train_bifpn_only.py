#!/usr/bin/env python3
"""
BiFPN独立实验 - 完整实现
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF
from utils.path_manager import path_manager


class BiFPN_Module(nn.Module):
    """BiFPN特征金字塔模块 - 完整实现"""

    def __init__(self, feature_channels=[256, 512, 1024], bifpn_channels=256):
        super(BiFPN_Module, self).__init__()

        self.feature_channels = feature_channels
        self.bifpn_channels = bifpn_channels
        self.num_levels = len(feature_channels)

        # 输入投影层
        self.input_proj = nn.ModuleList([
            nn.Conv2d(channels, bifpn_channels, 1) for channels in feature_channels
        ])

        # 自上而下路径 (Top-down path)
        self.top_down_convs = nn.ModuleList([
            nn.Sequential(
                Conv(bifpn_channels, bifpn_channels, 3, 1),
                nn.BatchNorm2d(bifpn_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_levels - 1)
        ])

        # 自下而上路径 (Bottom-up path)
        self.bottom_up_convs = nn.ModuleList([
            nn.Sequential(
                Conv(bifpn_channels, bifpn_channels, 3, 1),
                nn.BatchNorm2d(bifpn_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_levels - 1)
        ])

        # 输出投影层
        self.output_proj = nn.ModuleList([
            nn.Conv2d(bifpn_channels, channels, 1) for channels in feature_channels
        ])

        # 可学习的权重参数
        self.top_down_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(self.num_levels - 1)
        ])

        self.bottom_up_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(self.num_levels - 1)
        ])

        self.epsilon = 1e-4

    def weighted_fusion(self, features, weights):
        """加权特征融合"""
        normalized_weights = F.relu(weights)
        weight_sum = torch.sum(normalized_weights) + self.epsilon
        return sum(w / weight_sum * f for w, f in zip(normalized_weights, features))

    def forward(self, features):
        """
        前向传播
        features: 多尺度特征列表 [P3, P4, P5]
        """
        # 输入投影
        proj_features = []
        for i, feat in enumerate(features):
            proj_features.append(self.input_proj[i](feat))

        # 自上而下路径
        top_down_features = [proj_features[-1]]  # 从最高层开始

        for i in range(self.num_levels - 2, -1, -1):
            # 上采样并融合
            upsampled = F.interpolate(
                top_down_features[-1],
                scale_factor=2,
                mode='nearest'
            )

            fused = self.weighted_fusion(
                [proj_features[i], upsampled],
                self.top_down_weights[i]
            )

            top_down_features.append(self.top_down_convs[i](fused))

        # 反转顺序以匹配原始层级
        top_down_features = list(reversed(top_down_features))

        # 自下而上路径
        bottom_up_features = [top_down_features[0]]  # 从最底层开始

        for i in range(1, self.num_levels):
            # 下采样并融合
            downsampled = F.avg_pool2d(
                bottom_up_features[-1],
                kernel_size=3,
                stride=2,
                padding=1
            )

            fused = self.weighted_fusion(
                [top_down_features[i], downsampled],
                self.bottom_up_weights[i - 1]
            )

            bottom_up_features.append(self.bottom_up_convs[i - 1](fused))

        # 输出投影
        output_features = []
        for i, feat in enumerate(bottom_up_features):
            output_features.append(self.output_proj[i](feat) + features[i])  # 残差连接

        return output_features


class BiFPN_Model_Integrator:
    """BiFPN模型集成器"""

    def __init__(self):
        self.feature_layers = ['model.4', 'model.6', 'model.9']  # P3, P4, P5特征层
        self.detect_layer = 'model.22'  # Detect层

    def integrate_bifpn_into_model(self, model):
        """将BiFPN集成到模型中"""
        print("🔧 开始集成BiFPN特征金字塔到模型结构...")

        # 获取模型结构
        model_structure = model.model

        # 获取特征通道数
        feature_channels = self._get_feature_channels(model_structure)
        print(f"📊 特征通道数: {feature_channels}")

        # 创建BiFPN模块
        bifpn_module = BiFPN_Module(feature_channels)

        # 替换Detect层的forward方法
        success = self._replace_detect_forward(model_structure, bifpn_module)

        if success:
            print("✅ BiFPN集成成功")
            return model
        else:
            print("❌ BiFPN集成失败")
            return model

    def _get_feature_channels(self, model):
        """获取特征通道数"""
        feature_channels = []

        for layer_path in self.feature_layers:
            module = self._get_module_by_path(model, layer_path)
            if module is not None:
                # 获取输出通道数
                if hasattr(module, 'cv2') and hasattr(module.cv2, 'conv'):
                    channels = module.cv2.conv.out_channels
                elif hasattr(module, 'conv') and hasattr(module.conv, 'out_channels'):
                    channels = module.conv.out_channels
                else:
                    # 默认值
                    channels = 256

                feature_channels.append(channels)
                print(f"✅ {layer_path}: {channels} 通道")
            else:
                print(f"⚠ 找不到层: {layer_path}")
                feature_channels.append(256)  # 默认值

        return feature_channels

    def _replace_detect_forward(self, model, bifpn_module):
        """替换Detect层的forward方法以集成BiFPN"""
        try:
            # 查找Detect层
            detect_module = self._get_module_by_path(model, self.detect_layer)
            if detect_module is None:
                print("❌ 找不到Detect层")
                return False

            # 保存原始forward方法
            original_forward = detect_module.forward

            # 定义新的forward方法
            def new_forward(self, x):
                # 检查输入是否为多尺度特征
                if isinstance(x, (list, tuple)) and len(x) == 3:
                    # 应用BiFPN处理多尺度特征
                    bifpn_outputs = self.bifpn_module(x)
                    # 使用BiFPN输出
                    x = bifpn_outputs

                # 调用原始forward
                return original_forward(x)

            # 替换forward方法并添加属性
            detect_module.forward = new_forward.__get__(detect_module, type(detect_module))
            detect_module.bifpn_module = bifpn_module

            print("✅ Detect层forward方法替换成功")
            return True

        except Exception as e:
            print(f"❌ 替换Detect层forward方法失败: {e}")
            return False

    def _get_module_by_path(self, model, path):
        """根据路径获取模块"""
        try:
            modules = path.split('.')
            current_module = model

            for module_name in modules:
                if module_name.isdigit():
                    current_module = current_module[int(module_name)]
                else:
                    current_module = getattr(current_module, module_name)

            return current_module
        except (AttributeError, IndexError, KeyError):
            return None


class BiFPNOnlyExperiment:
    """BiFPN独立实验"""

    def __init__(self):
        self.exp_name = "bifpn_only"
        self.description = "YOLOv8s + BiFPN特征金字塔"
        self.integrator = BiFPN_Model_Integrator()

    def run(self):
        """运行BiFPN独立实验"""
        print("=" * 60)
        print("       BiFPN独立实验")
        print("=" * 60)
        print(f"实验名称: {self.exp_name}")
        print(f"描述: {self.description}")
        print("=" * 60)

        # 验证环境
        if not path_manager.validate_paths():
            print("❌ 环境验证失败")
            return False

        # 检查是否已训练
        exp_dir = path_manager.get_experiment_dir(self.exp_name)
        weights_file = exp_dir / "weights" / "best.pt"

        if weights_file.exists():
            print("✅ 实验已完成，跳过训练")
            return True

        # 加载基础模型
        print("🔄 加载YOLOv8s模型...")
        model = YOLO('yolov8s.pt')

        # 应用BiFPN改进
        print("🔧 应用BiFPN改进...")
        model = self.apply_bifpn_improvements(model)

        # 训练配置
        train_config = {
            'data': str(path_manager.dataset_config),
            'epochs': 80,
            'imgsz': 640,
            'batch': 16,
            'patience': 20,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 4,
            'project': str(path_manager.runs_dir),
            'name': self.exp_name,
            'exist_ok': True,
            'verbose': True,
            'save': True,
            'amp': False  # 关闭混合精度训练以确保稳定性
        }

        # 训练模型
        print("🚀 开始训练BiFPN模型...")
        try:
            results = model.train(**train_config)
            print("✅ BiFPN模型训练完成")
            return True
        except Exception as e:
            print(f"❌ BiFPN模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def apply_bifpn_improvements(self, model):
        """应用BiFPN改进"""
        print("🔧 开始集成BiFPN特征金字塔...")

        try:
            # 使用集成器修改模型结构
            model = self.integrator.integrate_bifpn_into_model(model)

            # 验证模型是否被修改
            bifpn_integrated = self._check_bifpn_integration(model.model)

            if bifpn_integrated:
                print("✅ BiFPN集成成功")
            else:
                print("❌ BiFPN集成失败")

            return model

        except Exception as e:
            print(f"❌ BiFPN集成失败: {e}")
            import traceback
            traceback.print_exc()
            return model  # 返回原始模型作为备选

    def _check_bifpn_integration(self, model):
        """检查BiFPN是否成功集成"""
        try:
            # 检查Detect层是否有bifpn_module属性
            detect_module = self.integrator._get_module_by_path(model, self.integrator.detect_layer)
            if detect_module is None:
                return False

            return hasattr(detect_module, 'bifpn_module')

        except Exception:
            return False

    def evaluate(self):
        """评估BiFPN模型"""
        print(f"\n📊 评估BiFPN模型...")

        exp_dir = path_manager.get_experiment_dir(self.exp_name)
        weights_file = exp_dir / "weights" / "best.pt"

        if not weights_file.exists():
            print("❌ 模型文件不存在")
            return None

        try:
            model = YOLO(str(weights_file))

            # 评估
            metrics = model.val(
                data=str(path_manager.dataset_config),
                split='val',
                imgsz=640,
                batch=16,
                conf=0.001,
                iou=0.6,
                device='cpu',
                verbose=False
            )

            # 提取指标
            result = {
                'map50': float(metrics.box.map50),
                'map': float(metrics.box.map),
                'precision': float(metrics.box.p.mean()),
                'recall': float(metrics.box.r.mean()),
                'f1_score': self.calculate_f1_score(
                    float(metrics.box.p.mean()),
                    float(metrics.box.r.mean())
                )
            }

            print(f"✅ BiFPN模型评估完成:")
            print(f"   mAP@0.5: {result['map50']:.4f}")
            print(f"   mAP@0.5:0.95: {result['map']:.4f}")
            print(f"   精确率: {result['precision']:.4f}")
            print(f"   召回率: {result['recall']:.4f}")
            print(f"   F1分数: {result['f1_score']:.4f}")

            return result

        except Exception as e:
            print(f"❌ 评估失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_f1_score(self, precision, recall):
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def diagnose_model(self):
        """诊断模型结构"""
        print(f"\n🔍 诊断BiFPN模型结构...")

        exp_dir = path_manager.get_experiment_dir(self.exp_name)
        weights_file = exp_dir / "weights" / "best.pt"

        if not weights_file.exists():
            print("❌ 模型文件不存在")
            return

        try:
            model = YOLO(str(weights_file))

            # 检查BiFPN集成
            bifpn_integrated = self._check_bifpn_integration(model.model)

            print(f"📊 模型诊断结果:")
            print(f"  BiFPN集成状态: {'✅ 成功' if bifpn_integrated else '❌ 失败'}")

            if bifpn_integrated:
                # 获取BiFPN模块
                detect_module = self.integrator._get_module_by_path(model.model, self.integrator.detect_layer)
                bifpn_module = detect_module.bifpn_module

                print(f"  BiFPN模块参数: {sum(p.numel() for p in bifpn_module.parameters()):,}")
                print(f"  输入通道: {bifpn_module.feature_channels}")
                print(f"  BiFPN通道: {bifpn_module.bifpn_channels}")
                print(f"  层级数量: {bifpn_module.num_levels}")

            print("✅ 模型诊断完成")

        except Exception as e:
            print(f"❌ 诊断失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    experiment = BiFPNOnlyExperiment()

    # 运行实验
    success = experiment.run()

    if success:
        # 诊断模型结构
        experiment.diagnose_model()

        # 评估模型
        experiment.evaluate()
        print("\n🎯 BiFPN独立实验完成!")
    else:
        print("\n❌ BiFPN独立实验失败")


if __name__ == "__main__":
    main()