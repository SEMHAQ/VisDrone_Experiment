#!/usr/bin/env python3
"""
EMA注意力独立实验 - 完整实现
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF
from utils.path_manager import path_manager


class EMAttention(nn.Module):
    """EMA注意力机制 - 完整实现"""

    def __init__(self, channels, reduction=16):
        super(EMAAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # 池化层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 卷积层
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # 归一化
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 水平池化
        x_h = self.pool_h(x)  # [b, c, h, 1]
        # 垂直池化
        x_w = self.pool_w(x)  # [b, c, 1, w]

        # 扩展维度以进行乘法
        x_h_expanded = x_h.expand(-1, -1, -1, width)
        x_w_expanded = x_w.expand(-1, -1, height, -1)

        # 计算注意力权重
        attention_weights = x_h_expanded * x_w_expanded
        attention_weights = self.conv(attention_weights)
        attention_weights = self.sigmoid(attention_weights)

        # 应用注意力权重
        output = x * attention_weights

        return output


class EMA_Bottleneck(nn.Module):
    """集成EMA注意力的Bottleneck模块"""

    def __init__(self, in_channels, out_channels, shortcut=True, groups=1):
        super(EMA_Bottleneck, self).__init__()
        self.cv1 = Conv(in_channels, out_channels, 1, 1)
        self.cv2 = Conv(out_channels, out_channels, 3, 1, groups=groups)
        self.ema_attention = EMAttention(out_channels)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.ema_attention(self.cv2(self.cv1(x))) if self.add else self.ema_attention(self.cv2(self.cv1(x)))


class EMA_C2f(nn.Module):
    """集成EMA注意力的C2f模块"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=False, groups=1, expansion=0.5):
        super(EMA_C2f, self).__init__()

        # 保持原始C2f结构
        hidden_dim = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_dim, 1, 1)
        self.conv2 = Conv((2 + n) * hidden_dim, out_channels, 1)

        # 创建EMA Bottleneck列表
        self.m = nn.ModuleList(EMA_Bottleneck(hidden_dim, hidden_dim, shortcut, groups) for _ in range(n))

        # 在输出前添加EMA注意力
        self.ema_attention = EMAttention(out_channels)

    def forward(self, x):
        # 原始C2f前向传播
        y = [self.conv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y.append(y[-1])  # 原始特征

        # 拼接并卷积
        x_out = self.conv2(torch.cat(y, 1))

        # 应用EMA注意力
        x_out = self.ema_attention(x_out)

        return x_out


class EMA_Model_Integrator:
    """EMA模型集成器"""

    def __init__(self):
        self.target_layers = {
            'backbone': ['model.4', 'model.6', 'model.9'],  # P3, P4, P5特征层
            'neck': ['model.13', 'model.17', 'model.20', 'model.23']  # Neck中的关键层
        }

    def integrate_ema_into_model(self, model):
        """将EMA注意力集成到模型中"""
        print("🔧 开始集成EMA注意力到模型结构...")

        # 获取模型结构
        model_structure = model.model

        # 替换关键层的C2f模块为EMA_C2f
        replaced_count = 0
        for layer_path in self.target_layers['backbone']:
            if self._replace_c2f_with_ema_c2f(model_structure, layer_path):
                replaced_count += 1

        for layer_path in self.target_layers['neck']:
            if self._replace_c2f_with_ema_c2f(model_structure, layer_path):
                replaced_count += 1

        print(f"✅ 成功替换了 {replaced_count} 个C2f模块为EMA_C2f")
        return model

    def _replace_c2f_with_ema_c2f(self, model, layer_path):
        """将指定路径的C2f模块替换为EMA_C2f"""
        try:
            # 获取目标模块
            target_module = self._get_module_by_path(model, layer_path)
            if target_module is None:
                print(f"⚠ 找不到层: {layer_path}")
                return False

            # 检查是否是C2f模块
            if not isinstance(target_module, C2f):
                print(f"⚠ {layer_path} 不是C2f模块，跳过")
                return False

            # 获取模块参数
            in_channels = target_module.cv1.conv.in_channels
            out_channels = target_module.cv2.conv.out_channels

            # 获取其他参数
            n = len(target_module.m) if hasattr(target_module, 'm') else 1
            shortcut = target_module.m[0].add if hasattr(target_module.m[0], 'add') else True

            # 创建EMA_C2f模块
            ema_c2f = EMA_C2f(in_channels, out_channels, n, shortcut)

            # 替换模块
            parent_module, module_name = self._get_parent_and_name(model, layer_path)
            if parent_module is not None:
                setattr(parent_module, module_name, ema_c2f)
                print(f"✅ 替换 {layer_path} 为EMA_C2f (输入: {in_channels}, 输出: {out_channels})")
                return True
            else:
                print(f"❌ 无法替换 {layer_path}")
                return False

        except Exception as e:
            print(f"❌ 替换 {layer_path} 失败: {e}")
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

    def _get_parent_and_name(self, model, path):
        """获取父模块和模块名称"""
        try:
            modules = path.split('.')
            if len(modules) == 1:
                return model, modules[0]

            parent_path = '.'.join(modules[:-1])
            module_name = modules[-1]

            parent_module = self._get_module_by_path(model, parent_path)
            return parent_module, module_name

        except Exception as e:
            print(f"❌ 获取父模块失败: {e}")
            return None, None


class EMAOnlyExperiment:
    """EMA注意力独立实验"""

    def __init__(self):
        self.exp_name = "ema_only"
        self.description = "YOLOv8s + EMA注意力机制"
        self.integrator = EMA_Model_Integrator()

    def run(self):
        """运行EMA独立实验"""
        print("=" * 60)
        print("       EMA注意力独立实验")
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

        # 应用EMA改进
        print("🔧 应用EMA注意力改进...")
        model = self.apply_ema_improvements(model)

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
        print("🚀 开始训练EMA模型...")
        try:
            results = model.train(**train_config)
            print("✅ EMA模型训练完成")
            return True
        except Exception as e:
            print(f"❌ EMA模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def apply_ema_improvements(self, model):
        """应用EMA注意力改进"""
        print("🔧 开始集成EMA注意力机制...")

        try:
            # 使用集成器修改模型结构
            model = self.integrator.integrate_ema_into_model(model)

            # 验证模型是否被修改
            ema_modules_count = self._count_ema_modules(model.model)
            print(f"✅ 模型修改完成，找到 {ema_modules_count} 个EMA模块")

            return model

        except Exception as e:
            print(f"❌ EMA集成失败: {e}")
            import traceback
            traceback.print_exc()
            return model  # 返回原始模型作为备选

    def _count_ema_modules(self, model):
        """统计模型中的EMA模块数量"""
        ema_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (EMA_C2f, EMA_Bottleneck, EMAttention)):
                ema_count += 1
        return ema_count

    def evaluate(self):
        """评估EMA模型"""
        print(f"\n📊 评估EMA模型...")

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

            print(f"✅ EMA模型评估完成:")
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
        print(f"\n🔍 诊断EMA模型结构...")

        exp_dir = path_manager.get_experiment_dir(self.exp_name)
        weights_file = exp_dir / "weights" / "best.pt"

        if not weights_file.exists():
            print("❌ 模型文件不存在")
            return

        try:
            model = YOLO(str(weights_file))

            # 统计EMA模块
            ema_count = self._count_ema_modules(model.model)
            print(f"📊 模型诊断结果:")
            print(f"  总EMA模块数量: {ema_count}")

            # 列出所有EMA模块
            print(f"  EMA模块位置:")
            for name, module in model.model.named_modules():
                if isinstance(module, (EMA_C2f, EMA_Bottleneck, EMAttention)):
                    print(f"    - {name}")

            if ema_count > 0:
                print("✅ EMA集成成功!")
            else:
                print("❌ EMA集成失败，模型未包含EMA模块")

        except Exception as e:
            print(f"❌ 诊断失败: {e}")


def main():
    """主函数"""
    experiment = EMAOnlyExperiment()

    # 运行实验
    success = experiment.run()

    if success:
        # 诊断模型结构
        experiment.diagnose_model()

        # 评估模型
        experiment.evaluate()
        print("\n🎯 EMA独立实验完成!")
    else:
        print("\n❌ EMA独立实验失败")


if __name__ == "__main__":
    main()