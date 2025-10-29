import torch
import torch.nn as nn
from ultralytics import YOLO
import types
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# 首先定义 EMAttention 类，确保它在全局作用域中可用
class EMAttention(nn.Module):
    """EMA注意力机制 - 简化版本"""

    def __init__(self, channels, reduction=16):
        # 使用 super() 而不带类名
        super().__init__()
        self.groups = max(1, channels // reduction)

        # 池化层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 卷积层
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 水平池化
        x_h = self.pool_h(x)  # [b, c, h, 1]
        # 垂直池化
        x_w = self.pool_w(x)  # [b, c, 1, w]

        # 计算注意力权重
        x_h_expanded = x_h.expand(-1, -1, -1, w)
        x_w_expanded = x_w.expand(-1, -1, h, -1)

        attention = self.sigmoid(self.conv(x_h_expanded * x_w_expanded))

        return x * attention


# 定义 BiFPN_Module 类
class BiFPN_Module(nn.Module):
    """BiFPN模块 - 替换YOLOv8的PANet"""

    def __init__(self, feature_channels=[256, 512, 1024], bifpn_channels=256):
        super().__init__()

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


class HookManager:
    """Hook管理器"""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.original_methods = {}

    def register_hook(self, module, hook_type, hook_fn):
        """注册hook"""
        if hook_type == 'forward':
            handle = module.register_forward_hook(hook_fn)
        elif hook_type == 'forward_pre':
            handle = module.register_forward_pre_hook(hook_fn)
        else:
            raise ValueError(f"不支持的hook类型: {hook_type}")

        self.hooks.append(handle)
        return handle

    def remove_hooks(self):
        """移除所有hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class YOLOv8HookIntegrator:
    """YOLOv8 Hook集成器 - 简化版本"""

    def __init__(self, model_path='yolov8s.pt'):
        self.model = YOLO(model_path)
        self.hook_manager = HookManager(self.model.model)
        self.ema_hooks = []

    def integrate_ema_attention(self, target_layers=None, reduction=16):
        """集成EMA注意力机制 - 修复版本"""
        print("🔧 集成EMA注意力机制...")

        if target_layers is None:
            # 使用更简单的层路径
            target_layers = ['model.4', 'model.6', 'model.9']

        for layer_path in target_layers:
            try:
                module = self._get_module_by_path(layer_path)
                if module is None:
                    print(f"⚠ 找不到层: {layer_path}")
                    continue

                # 获取通道数
                if hasattr(module, 'cv2') and hasattr(module.cv2, 'conv'):
                    channels = module.cv2.conv.out_channels
                else:
                    # 估计通道数
                    channels = 256
                    print(f"⚠ 无法确定 {layer_path} 的通道数，使用默认值: {channels}")

                # 创建EMA注意力实例
                ema_layer = EMAttention(channels, reduction)

                # 创建hook函数
                def create_ema_hook(ema_instance):
                    def ema_hook(module, input, output):
                        return ema_instance(output)

                    return ema_hook

                # 注册hook
                hook_fn = create_ema_hook(ema_layer)
                handle = self.hook_manager.register_hook(module, 'forward', hook_fn)
                self.ema_hooks.append((layer_path, handle))

                print(f"✅ 在 {layer_path} 添加EMA注意力 (通道数: {channels})")

            except Exception as e:
                print(f"❌ 在 {layer_path} 添加EMA失败: {e}")

        return len(self.ema_hooks)

    def integrate_bifpn(self, feature_layer_paths=None):
        """集成BiFPN特征金字塔"""
        print("🔧 集成BiFPN特征金字塔...")

        if feature_layer_paths is None:
            # 默认特征层路径 (P3, P4, P5)
            feature_layer_paths = ['model.4', 'model.6', 'model.9']

        # 获取特征通道数
        feature_channels = []
        for path in feature_layer_paths:
            module = self._get_module_by_path(path)
            if module and hasattr(module, 'cv2'):
                feature_channels.append(module.cv2.conv.out_channels)
            else:
                feature_channels.append(256)  # 默认值

        # 创建BiFPN模块
        self.bifpn_module = BiFPN_Module(feature_channels)

        # 注册特征捕获hook
        for i, path in enumerate(feature_layer_paths):
            module = self._get_module_by_path(path)
            if module:
                def create_capture_hook(idx):
                    def capture_hook(module, input, output):
                        return output

                    return capture_hook

                self.hook_manager.register_hook(
                    module, 'forward', create_capture_hook(i)
                )

        print("✅ BiFPN集成完成")
        return True

    def integrate_full_model(self):
        """集成完整模型（EMA + BiFPN）"""
        print("🔧 集成完整模型...")

        # 集成EMA
        ema_count = self.integrate_ema_attention()
        print(f"✅ 添加了 {ema_count} 个EMA注意力模块")

        # 集成BiFPN
        bifpn_success = self.integrate_bifpn()
        print(f"✅ BiFPN集成: {bifpn_success}")

        return ema_count > 0 and bifpn_success

    def _get_module_by_path(self, path):
        """根据路径获取模块"""
        try:
            modules = path.split('.')
            current_module = self.model.model

            for module_name in modules:
                if module_name.isdigit():
                    current_module = current_module[int(module_name)]
                else:
                    current_module = getattr(current_module, module_name)

            return current_module
        except (AttributeError, IndexError, KeyError):
            return None

    def train(self, **kwargs):
        """训练模型"""
        print("🚀 开始训练集成模型...")
        try:
            results = self.model.train(**kwargs)
            print("✅ 训练完成")
            return results
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            return None

    def cleanup(self):
        """清理hook"""
        self.hook_manager.remove_hooks()
        print("🧹 清理hook完成")


def create_ema_model():
    """创建集成EMA注意力的模型"""
    integrator = YOLOv8HookIntegrator('yolov8s.pt')
    integrator.integrate_ema_attention()
    return integrator


def create_bifpn_model():
    """创建集成BiFPN的模型"""
    integrator = YOLOv8HookIntegrator('yolov8s.pt')
    integrator.integrate_bifpn()
    return integrator


def create_full_model():
    """创建集成EMA和BiFPN的完整模型"""
    integrator = YOLOv8HookIntegrator('yolov8s.pt')
    integrator.integrate_full_model()
    return integrator


# 测试函数
def test_hook_integration():
    """测试Hook集成"""
    print("🧪 测试Hook集成...")

    # 测试EMA集成
    print("\n1. 测试EMA集成:")
    ema_integrator = create_ema_model()
    print(f"EMA hooks数量: {len(ema_integrator.ema_hooks)}")

    # 测试BiFPN集成
    print("\n2. 测试BiFPN集成:")
    bifpn_integrator = create_bifpn_model()
    print(f"BiFPN集成: {bifpn_integrator.bifpn_module is not None}")

    # 测试完整集成
    print("\n3. 测试完整集成:")
    full_integrator = create_full_model()
    print(f"EMA hooks数量: {len(full_integrator.ema_hooks)}")
    print(f"BiFPN集成: {full_integrator.bifpn_module is not None}")

    # 清理
    ema_integrator.cleanup()
    bifpn_integrator.cleanup()
    full_integrator.cleanup()

    print("✅ Hook集成测试完成")


if __name__ == "__main__":
    test_hook_integration()