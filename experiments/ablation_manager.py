import os
import yaml
from pathlib import Path


class AblationManager:
    """消融实验管理器 - 完整版本"""

    def __init__(self):
        self.experiments = {
            'baseline': {
                'description': '基准模型 - 原始YOLOv8s',
                'model_class': 'BaselineModel',
                'modules': [],
                'config_file': 'baseline.yaml'
            },
            'ema_attention': {
                'description': '基准模型 + EMA注意力机制',
                'model_class': 'EMAModel',
                'modules': ['ema_attention'],
                'config_file': 'ema_attention.yaml'
            },
            'bifpn': {
                'description': '基准模型 + BiFPN特征金字塔',
                'model_class': 'BiFPNModel',
                'modules': ['bifpn'],
                'config_file': 'bifpn.yaml'
            },
            'full_model': {
                'description': '完整模型 - 所有改进组合',
                'model_class': 'FullModel',
                'modules': ['image_enhance', 'ema_attention', 'bifpn'],
                'config_file': 'full_model.yaml'
            }
        }

    def setup_all_experiments(self):
        """设置所有实验配置"""
        print("🚀 设置消融实验框架")

        # 确保配置目录存在
        config_dir = Path("configs/experiments")
        config_dir.mkdir(parents=True, exist_ok=True)

        # 创建所有实验配置
        for exp_name, config in self.experiments.items():
            self._create_experiment_config(exp_name, config)

        print("✅ 所有实验配置创建完成")
        self.get_experiment_status()

    def _create_experiment_config(self, exp_name, config):
        """创建单个实验配置"""
        config_path = Path(f"configs/experiments/{exp_name}.yaml")

        # 基础配置
        base_config = {
            'experiment_name': exp_name,
            'description': config['description'],
            'model_class': config['model_class'],
            'base_model': 'yolov8s.pt',
            'dataset_config': 'configs/dataset/visdrone.yaml',
            'epochs': 80,
            'imgsz': 640,
            'batch_size': 16,
            'patience': 20,
            'output_dir': f'runs/{exp_name}',
            'enabled_modules': config['modules'],
            'module_config': self._get_module_config(config['modules'])
        }

        # 保存配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)

        print(f"   ✅ {exp_name}: {config_path}")

    def _get_module_config(self, modules):
        """根据启用的模块生成配置"""
        module_config = {}

        if 'ema_attention' in modules:
            module_config.update({
                'ema_attention': True,
                'attention_type': 'EMA',
                'attention_channels': 512
            })

        if 'bifpn' in modules:
            module_config.update({
                'bifpn': True,
                'bifpn_channels': 256,
                'bifpn_levels': 5
            })

        if 'image_enhance' in modules:
            module_config.update({
                'image_enhance': True,
                'enhancement_methods': ['clahe', 'deblur']
            })

        return module_config

    def get_experiment_status(self):
        """获取实验状态"""
        print("\n📊 实验状态检查")
        print("=" * 50)

        for exp_name, config in self.experiments.items():
            config_file = Path(f"configs/experiments/{exp_name}.yaml")
            output_dir = Path(f"runs/{exp_name}")

            config_status = "✅" if config_file.exists() else "❌"
            output_status = "✅" if output_dir.exists() else "❌"

            print(f"{config_status} {output_status} {exp_name}: {config['description']}")


# 全局管理器实例
ablation_manager = AblationManager()