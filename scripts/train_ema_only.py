#!/usr/bin/env python3
"""
EMA注意力独立实验 - 修正版（兼容YOLOv8原结构）
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


# =====================================================
# ================  EMA 注意力模块  ====================
# =====================================================
class EMAttention(nn.Module):
    """EMA注意力机制"""

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


# =====================================================
# ================  EMA Bottleneck ====================
# =====================================================
class EMA_Bottleneck(nn.Module):
    """集成EMA注意力的Bottleneck模块（兼容YOLOv8原结构参数 g 而非 groups）"""

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


# =====================================================
# ================  EMA_C2f模块 =======================
# =====================================================
class EMA_C2f(nn.Module):
    """集成EMA注意力的C2f模块（修正版）"""

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


# =====================================================
# ================  EMA模型集成器  ====================
# =====================================================
class EMA_Model_Integrator:
    """EMA模型集成器（修正版）"""

    def __init__(self):
        self.target_layers = ['model.2', 'model.4', 'model.6', 'model.8']

    def integrate_ema_into_model(self, model):
        print("🔧🔧 开始集成EMA注意力到模型结构...")
        model_structure = model.model
        print("📋📋 模型结构分析:")
        self._analyze_model_structure(model_structure)

        replaced_count = 0
        for layer_path in self.target_layers:
            if self._replace_c2f_with_ema_c2f(model_structure, layer_path):
                replaced_count += 1

        print(f"✅ 成功替换了 {replaced_count} 个C2f模块为EMA_C2f")
        return model

    def _analyze_model_structure(self, model):
        print("🔍🔍 分析模型层结构...")
        model_to_analyze = model.model if hasattr(model, 'model') else model
        if not hasattr(model_to_analyze, '__iter__'):
            print("⚠ 模型结构不可迭代，跳过详细分析")
            return

        for i, module in enumerate(model_to_analyze):
            module_type = type(module).__name__
            print(f"层 {i}: {module_type}")
            if isinstance(module, C2f):
                print(f"  → C2f模块: 输入={module.cv1.conv.in_channels}, 输出={module.cv2.conv.out_channels}")

    def _replace_c2f_with_ema_c2f(self, model, layer_path):
        try:
            target_module = self._get_module_by_path(model, layer_path)
            if target_module is None:
                print(f"⚠ 找不到层: {layer_path}")
                return False
            if not isinstance(target_module, C2f):
                print(f"⚠ {layer_path} 不是C2f模块，实际是: {type(target_module).__name__}")
                return False

            # 获取参数
            c1 = target_module.cv1.conv.in_channels
            c2 = target_module.cv2.conv.out_channels
            n = len(target_module.m)
            shortcut = getattr(target_module.m[0], "add", False)
            g = getattr(target_module.m[0].cv2.conv, "groups", 1)
            e = getattr(target_module, "e", 0.5)

            ema_c2f = EMA_C2f(c1, c2, n=n, shortcut=shortcut, g=g, e=e)

            parent_module, module_name = self._get_parent_and_name(model, layer_path)
            if parent_module is not None:
                setattr(parent_module, module_name, ema_c2f)
                print(f"✅ 替换 {layer_path} 为EMA_C2f (输入: {c1}, 输出: {c2})")
                return True
            return False
        except Exception as e:
            print(f"❌❌ 替换 {layer_path} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_module_by_path(self, model, path):
        try:
            modules = path.split(".")
            current = model
            for m in modules:
                current = current[int(m)] if m.isdigit() else getattr(current, m)
            return current
        except Exception:
            return None

    def _get_parent_and_name(self, model, path):
        modules = path.split(".")
        parent_path = ".".join(modules[:-1])
        module_name = modules[-1]
        parent_module = self._get_module_by_path(model, parent_path)
        return parent_module, module_name


# =====================================================
# ================  实验主体逻辑 ======================
# =====================================================
class EMAOnlyExperiment:
    """EMA注意力独立实验"""

    def __init__(self):
        self.exp_name = "ema_only_fixed"
        self.description = "YOLOv8s + EMA注意力机制"
        self.integrator = EMA_Model_Integrator()

    def run(self):
        print("=" * 60)
        print("       EMA注意力独立实验（修正版）")
        print("=" * 60)
        if not path_manager.validate_paths():
            print("❌ 环境验证失败")
            return False

        exp_dir = path_manager.get_experiment_dir(self.exp_name)
        weights_file = exp_dir / "weights" / "best.pt"
        if weights_file.exists():
            print("✅ 已存在训练权重，跳过训练")
            return True

        print("🔄 加载YOLOv8s模型...")
        model = YOLO("yolov8s.pt")

        print("🔧 集成EMA模块...")
        model = self.integrator.integrate_ema_into_model(model)
        print("📊 模型结构集成完毕")

        train_config = {
            'data': str(path_manager.dataset_config),
            'epochs': 50,
            'imgsz': 640,
            'batch': 8,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 2,
            'project': str(path_manager.runs_dir),
            'name': self.exp_name,
            'exist_ok': True,
            'amp': False,
        }

        print("🚀 开始训练EMA模型...")
        model.train(**train_config)
        print("✅ EMA模型训练完成")
        return True


def main():
    experiment = EMAOnlyExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
