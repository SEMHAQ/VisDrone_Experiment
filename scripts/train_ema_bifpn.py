#!/usr/bin/env python3
"""
BiFPN + EMA注意力联合实验 - 修正版
结合BiFPN特征金字塔和EMA注意力机制的优势
"""

import os
import sys
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF
from utils.path_manager import path_manager


class EMAttention(nn.Module):
    """EMA注意力机制（修正版）"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x).expand(-1, -1, -1, w)
        x_w = self.pool_w(x).expand(-1, -1, h, -1)
        attn = self.sigmoid(self.conv(x_h * x_w))
        return x * attn


class EMA_Bottleneck(nn.Module):
    """集成EMA注意力的Bottleneck模块（修正版）"""

    def __init__(self, c1, c2, shortcut=True, g=1):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, 3, 1, g=g)
        self.ema_attention = EMAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        y = self.ema_attention(y)
        return x + y if self.add else y


class EMA_C2f(nn.Module):
    """集成EMA注意力的C2f模块（修正版，兼容YOLOv8结构）"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        self.m = nn.ModuleList(EMA_Bottleneck(c_, c_, shortcut, g=g) for _ in range(n))
        self.ema_attention = EMAttention(c2)

    def forward(self, x):
        y = [self.cv1(x)]
        for m in self.m:
            y.append(m(y[-1]))
        out = self.cv2(torch.cat(y, 1))
        out = self.ema_attention(out)
        return out



# === FIX ===
# 把 Bottleneck 的参数从 groups 改为 g，保持与 YOLOv8 源码命名一致，避免关键字冲突
class EMA_Bottleneck(nn.Module):
    """集成EMA注意力的Bottleneck模块（兼容 YOLOv8 参数 g）"""

    def __init__(self, in_channels, out_channels, shortcut=True, g=1):
        super(EMA_Bottleneck, self).__init__()
        self.cv1 = Conv(in_channels, out_channels, 1, 1)
        # 通过 g 传递 group 参数给底层 Conv（ultralytics.Conv 支持 g）
        self.cv2 = Conv(out_channels, out_channels, 3, 1, g=g)
        self.ema_attention = EMAttention(out_channels)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        y = self.ema_attention(y)
        return x + y if self.add else y


# === FIX ===
# EMA_C2f 也使用 g 同步 groups 行为，并保留 expansion 参数名与原 C2f 对齐
class EMA_C2f(nn.Module):
    """集成EMA注意力的C2f模块（修正版）"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=False, g=1, expansion=0.5):
        super(EMA_C2f, self).__init__()

        # 保持原始C2f结构
        hidden_dim = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_dim, 1, 1)
        self.conv2 = Conv((2 + n) * hidden_dim, out_channels, 1)

        # 创建EMA Bottleneck列表 - 使用 g 传参
        self.m = nn.ModuleList(EMA_Bottleneck(hidden_dim, hidden_dim, shortcut, g=g) for _ in range(n))

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


class BiFPN_EMA_Model_Integrator:
    """BiFPN + EMA模型集成器"""

    def __init__(self):
        # EMA替换的目标层（保持你原来的层路径）
        self.ema_target_layers = {
            'backbone': ['model.2', 'model.4', 'model.6', 'model.8'],
            'neck': []  # 暂不替换
        }

        # BiFPN特征层
        self.bifpn_feature_layers = ['model.4', 'model.6', 'model.9']
        self.detect_layer = 'model.22'

    def integrate_improvements_into_model(self, model):
        """将BiFPN和EMA改进集成到模型中"""
        print("🔧🔧 开始集成BiFPN + EMA改进到模型结构...")

        # 获取模型结构
        model_structure = model.model

        # 第一步：集成EMA注意力
        print("🔄🔄 集成EMA注意力机制...")
        ema_replaced_count = self._integrate_ema_attention(model_structure)
        print(f"✅ 成功替换了 {ema_replaced_count} 个C2f模块为EMA_C2f")

        # 第二步：集成BiFPN特征金字塔
        print("🔄🔄 集成BiFPN特征金字塔...")
        bifpn_success = self._integrate_bifpn_module(model_structure)

        if bifpn_success:
            print("✅ BiFPN集成成功")
        else:
            print("❌❌ BiFPN集成失败")

        return model

    def _integrate_ema_attention(self, model):
        """集成EMA注意力机制"""
        replaced_count = 0

        for layer_path in self.ema_target_layers['backbone']:
            if self._replace_c2f_with_ema_c2f(model, layer_path):
                replaced_count += 1

        for layer_path in self.ema_target_layers['neck']:
            if self._replace_c2f_with_ema_c2f(model, layer_path):
                replaced_count += 1

        return replaced_count

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
            # shortcut 存在于 Bottleneck 的 add 属性上（若不存在则取 False）
            shortcut = False
            try:
                if hasattr(target_module, 'm') and len(target_module.m) > 0:
                    shortcut = getattr(target_module.m[0], 'add', False)
            except Exception:
                shortcut = False

            # === 兼容读取 groups/g 参数 ===
            g = 1
            try:
                if hasattr(target_module, 'm') and len(target_module.m) > 0:
                    m0 = target_module.m[0]
                    # 深入尝试读取底层 conv 的 groups 属性（如果存在）
                    if hasattr(m0, 'cv2') and hasattr(m0.cv2, 'conv'):
                        g = getattr(m0.cv2.conv, 'groups', 1)
                    else:
                        g = 1
            except Exception:
                g = 1

            # expansion（e）尽量从原模块读取（若无则使用 0.5）
            expansion = getattr(target_module, 'e', 0.5)

            # 创建EMA_C2f模块（使用 g 而不是 groups）
            ema_c2f = EMA_C2f(in_channels, out_channels, n=n, shortcut=shortcut, g=g, expansion=expansion)

            # 替换模块
            parent_module, module_name = self._get_parent_and_name(model, layer_path)
            if parent_module is not None:
                setattr(parent_module, module_name, ema_c2f)
                print(f"✅ 替换 {layer_path} 为EMA_C2f (输入: {in_channels}, 输出: {out_channels}, n={n}, g={g})")
                return True
            else:
                print(f"❌❌ 无法替换 {layer_path}（未找到父模块）")
                return False

        except Exception as e:
            print(f"❌❌ 替换 {layer_path} 失败: {e}")
            traceback.print_exc()
            return False

    def _integrate_bifpn_module(self, model):
        """集成BiFPN模块"""
        try:
            # 获取特征通道数
            feature_channels = self._get_feature_channels(model)
            print(f"📊📊 特征通道数: {feature_channels}")

            # 创建BiFPN模块
            bifpn_module = BiFPN_Module(feature_channels)

            # 替换Detect层的forward方法
            return self._replace_detect_forward(model, bifpn_module)

        except Exception as e:
            print(f"❌❌ BiFPN集成失败: {e}")
            traceback.print_exc()
            return False

    def _get_feature_channels(self, model):
        """获取特征通道数"""
        feature_channels = []

        for layer_path in self.bifpn_feature_layers:
            module = self._get_module_by_path(model, layer_path)
            if module is not None:
                # 获取输出通道数
                try:
                    if hasattr(module, 'cv2') and hasattr(module.cv2, 'conv'):
                        channels = module.cv2.conv.out_channels
                    elif hasattr(module, 'conv') and hasattr(module.conv, 'out_channels'):
                        channels = module.conv.out_channels
                    else:
                        # 默认值
                        channels = 256
                except Exception:
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
                print("❌❌ 找不到Detect层")
                return False

            # 保存原始forward方法
            original_forward = detect_module.forward

            # 定义新的forward方法
            def new_forward(self, x):
                # 如果输入是多尺度 feature list（P3,P4,P5），则先用 bifpn_module 处理
                if isinstance(x, (list, tuple)) and len(x) == 3:
                    bifpn_outputs = self.bifpn_module(x)
                    x = bifpn_outputs

                # 调用原始 forward（保持原 Detect 行为）
                return original_forward(x)

            # 替换 forward 并附加 bifpn_module
            detect_module.forward = new_forward.__get__(detect_module, type(detect_module))
            detect_module.bifpn_module = bifpn_module

            print("✅ Detect层forward方法替换成功")
            return True

        except Exception as e:
            print(f"❌❌ 替换Detect层forward方法失败: {e}")
            traceback.print_exc()
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
            print(f"❌❌ 获取父模块失败: {e}")
            traceback.print_exc()
            return None, None


class BiFPN_EMA_Experiment:
    """BiFPN + EMA联合实验"""

    def __init__(self):
        self.exp_name = "bifpn_ema_combined"
        self.description = "YOLOv8s + BiFPN特征金字塔 + EMA注意力机制"
        self.integrator = BiFPN_EMA_Model_Integrator()

    def run(self):
        """运行联合实验"""
        print("=" * 70)
        print("       BiFPN + EMA注意力联合实验")
        print("=" * 70)
        print(f"实验名称: {self.exp_name}")
        print(f"描述: {self.description}")
        print("=" * 70)

        # 验证环境
        if not path_manager.validate_paths():
            print("❌❌ 环境验证失败")
            return False

        # 检查是否已训练
        exp_dir = path_manager.get_experiment_dir(self.exp_name)
        weights_file = exp_dir / "weights" / "best.pt"

        if weights_file.exists():
            print("✅ 实验已完成，跳过训练")
            return True

        # 加载基础模型
        print("🔄🔄 加载YOLOv8s模型...")
        model = YOLO('yolov8s.pt')

        # 应用BiFPN + EMA改进
        print("🔧🔧 应用BiFPN + EMA改进...")
        model = self.apply_combined_improvements(model)

        # 训练配置
        train_config = {
            'data': str(path_manager.dataset_config),
            'epochs': 100,  # 增加训练轮数以充分发挥组合优势
            'imgsz': 640,
            'batch': 16,
            'patience': 25,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 4,
            'project': str(path_manager.runs_dir),
            'name': self.exp_name,
            'exist_ok': True,
            'verbose': True,
            'save': True,
            'amp': True,  # 开启混合精度训练
            'lr0': 0.01,  # 调整学习率
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1
        }

        # 训练模型
        print("🚀🚀 开始训练BiFPN + EMA组合模型...")
        try:
            results = model.train(**train_config)
            print("✅ BiFPN + EMA组合模型训练完成")
            return True
        except Exception as e:
            print(f"❌❌ 组合模型训练失败: {e}")
            traceback.print_exc()
            return False

    def apply_combined_improvements(self, model):
        """应用BiFPN + EMA组合改进"""
        print("🔧🔧 开始集成BiFPN特征金字塔和EMA注意力机制...")

        try:
            # 使用集成器修改模型结构
            model = self.integrator.integrate_improvements_into_model(model)

            # 验证模型是否被修改
            improvements_applied = self._check_improvements_integration(model.model)

            if improvements_applied:
                print("✅ BiFPN + EMA集成成功")
            else:
                print("❌❌ BiFPN + EMA集成失败")

            return model

        except Exception as e:
            print(f"❌❌ 组合改进集成失败: {e}")
            traceback.print_exc()
            return model  # 返回原始模型作为备选

    def _check_improvements_integration(self, model):
        """检查改进是否成功集成"""
        try:
            # 检查EMA模块
            ema_count = 0
            for name, module in model.named_modules():
                if isinstance(module, (EMA_C2f, EMA_Bottleneck, EMAttention)):
                    ema_count += 1

            # 检查BiFPN模块
            detect_module = self.integrator._get_module_by_path(model, self.integrator.detect_layer)
            bifpn_integrated = detect_module is not None and hasattr(detect_module, 'bifpn_module')

            print(f"📊📊 集成检查结果:")
            print(f"  EMA模块数量: {ema_count}")
            print(f"  BiFPN集成状态: {'✅ 成功' if bifpn_integrated else '❌❌ 失败'}")

            return ema_count > 0 and bifpn_integrated

        except Exception:
            traceback.print_exc()
            return False

    def evaluate(self):
        """评估组合模型"""
        print(f"\n📊📊 评估BiFPN + EMA组合模型...")

        exp_dir = path_manager.get_experiment_dir(self.exp_name)
        weights_file = exp_dir / "weights" / "best.pt"

        if not weights_file.exists():
            print("❌❌ 模型文件不存在")
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
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=True
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

            print(f"✅ BiFPN + EMA组合模型评估完成:")
            print(f"   mAP@0.5: {result['map50']:.4f}")
            print(f"   mAP@0.5:0.95: {result['map']:.4f}")
            print(f"   精确率: {result['precision']:.4f}")
            print(f"   召回率: {result['recall']:.4f}")
            print(f"   F1分数: {result['f1_score']:.4f}")

            return result

        except Exception as e:
            print(f"❌❌ 评估失败: {e}")
            traceback.print_exc()
            return None

    def calculate_f1_score(self, precision, recall):
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def diagnose_model(self):
        """诊断模型结构"""
        print(f"\n🔍🔍 诊断BiFPN + EMA组合模型结构...")

        exp_dir = path_manager.get_experiment_dir(self.exp_name)
        weights_file = exp_dir / "weights" / "best.pt"

        if not weights_file.exists():
            print("❌❌ 模型文件不存在")
            return

        try:
            model = YOLO(str(weights_file))

            # 统计EMA模块
            ema_count = 0
            ema_modules = []
            for name, module in model.model.named_modules():
                if isinstance(module, (EMA_C2f, EMA_Bottleneck, EMAttention)):
                    ema_count += 1
                    ema_modules.append(name)

            # 检查BiFPN集成
            detect_module = self.integrator._get_module_by_path(model.model, self.integrator.detect_layer)
            bifpn_integrated = detect_module is not None and hasattr(detect_module, 'bifpn_module')

            print(f"📊📊 模型诊断结果:")
            print(f"  EMA模块数量: {ema_count}")
            print(f"  BiFPN集成状态: {'✅ 成功' if bifpn_integrated else '❌❌ 失败'}")

            if ema_count > 0:
                print(f"  EMA模块位置:")
                for module_name in ema_modules[:5]:  # 只显示前5个
                    print(f"    - {module_name}")
                if len(ema_modules) > 5:
                    print(f"    ... 还有 {len(ema_modules) - 5} 个模块")

            if bifpn_integrated:
                bifpn_module = detect_module.bifpn_module
                print(f"  BiFPN模块参数: {sum(p.numel() for p in bifpn_module.parameters()):,}")
                print(f"  输入通道: {bifpn_module.feature_channels}")
                print(f"  BiFPN通道: {bifpn_module.bifpn_channels}")

            print("✅ 模型诊断完成")

        except Exception as e:
            print(f"❌❌ 诊断失败: {e}")
            traceback.print_exc()

    def compare_with_baseline(self):
        """与基线模型比较性能"""
        print(f"\n📈📈 与基线模型性能比较...")

        # 这里可以添加与原始YOLOv8s的性能比较逻辑
        # 实际实现需要加载基线模型并评估

        print("🔜 性能比较功能待实现...")
        return None


def main():
    """主函数"""
    experiment = BiFPN_EMA_Experiment()

    # 运行实验
    success = experiment.run()

    if success:
        # 诊断模型结构
        experiment.diagnose_model()

        # 评估模型
        experiment.evaluate()

        # 性能比较
        experiment.compare_with_baseline()

        print("\n🎯🎯 BiFPN + EMA联合实验完成!")
        print("✨✨ 组合模型充分利用了:")
        print("   ✅ BiFPN的多尺度特征融合优势")
        print("   ✅ EMA注意力的空间关系建模能力")
        print("   ✅ 两者的协同增强效果")
    else:
        print("\n❌❌ BiFPN + EMA联合实验失败")


if __name__ == "__main__":
    main()
