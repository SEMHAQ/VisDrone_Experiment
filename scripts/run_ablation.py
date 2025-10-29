#!/usr/bin/env python3
"""
消融实验运行脚本 - 完整版本
"""

import os
import sys
import yaml
from pathlib import Path
import subprocess
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.path_manager import path_manager


class AblationRunner:
    """消融实验运行器"""

    def __init__(self):
        self.experiments = {
            'baseline': {
                'description': '基准模型 - 原始YOLOv8s',
                'config_file': 'configs/experiments/baseline.yaml',
                'status': 'pending'  # pending, running, completed, failed
            },
            'ema_attention': {
                'description': '基准模型 + EMA注意力机制',
                'config_file': 'configs/experiments/ema_attention.yaml',
                'status': 'pending'
            },
            'bifpn': {
                'description': '基准模型 + BiFPN特征金字塔',
                'config_file': 'configs/experiments/bifpn.yaml',
                'status': 'pending'
            },
            'full_model': {
                'description': '完整模型 - 所有改进组合',
                'config_file': 'configs/experiments/full_model.yaml',
                'status': 'pending'
            }
        }

    def setup_experiments(self):
        """设置所有实验配置"""
        print("🛠 设置消融实验配置...")

        # 确保配置目录存在
        config_dir = Path("configs/experiments")
        config_dir.mkdir(parents=True, exist_ok=True)

        # 创建实验配置
        for exp_name, config in self.experiments.items():
            self._create_experiment_config(exp_name, config)

        print("✅ 所有实验配置创建完成")

    def _create_experiment_config(self, exp_name, config):
        """创建单个实验配置"""
        config_path = Path(config['config_file'])

        # 基础配置
        base_config = {
            'experiment_name': exp_name,
            'description': config['description'],
            'model_class': f"{exp_name.capitalize()}Model",
            'base_model': 'yolov8s.pt',
            'dataset_config': 'configs/dataset/visdrone.yaml',
            'epochs': 80,
            'imgsz': 640,
            'batch_size': 16,
            'patience': 20,
            'output_dir': f'runs/{exp_name}',
            'enabled_modules': self._get_enabled_modules(exp_name),
            'module_config': self._get_module_config(exp_name)
        }

        # 保存配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)

        print(f"   ✅ {exp_name}: {config_path}")

    def _get_enabled_modules(self, exp_name):
        """根据实验名称获取启用的模块"""
        modules_map = {
            'baseline': [],
            'ema_attention': ['ema_attention'],
            'bifpn': ['bifpn'],
            'full_model': ['image_enhance', 'ema_attention', 'bifpn']
        }
        return modules_map.get(exp_name, [])

    def _get_module_config(self, exp_name):
        """根据实验名称获取模块配置"""
        base_config = {}

        if exp_name in ['ema_attention', 'full_model']:
            base_config.update({
                'ema_attention': True,
                'attention_type': 'EMA',
                'attention_channels': 512
            })

        if exp_name in ['bifpn', 'full_model']:
            base_config.update({
                'bifpn': True,
                'bifpn_channels': 256,
                'bifpn_levels': 5
            })

        if exp_name == 'full_model':
            base_config.update({
                'image_enhance': True,
                'enhancement_methods': ['clahe', 'deblur']
            })

        return base_config

    def check_experiment_status(self):
        """检查实验状态"""
        print("\n📊 实验状态检查")
        print("=" * 50)

        for exp_name, config in self.experiments.items():
            output_dir = Path(f"runs/{exp_name}")
            weights_file = output_dir / "weights" / "best.pt"

            if weights_file.exists():
                config['status'] = 'completed'
                status_icon = "✅"
            elif output_dir.exists():
                config['status'] = 'running'
                status_icon = "🔄"
            else:
                config['status'] = 'pending'
                status_icon = "⏳"

            print(f"{status_icon} {exp_name}: {config['description']}")

        return self.experiments

    def run_single_experiment(self, exp_name):
        """运行单个实验"""
        print(f"\n🚀 开始运行实验: {exp_name}")
        print("=" * 50)

        config = self.experiments[exp_name]
        print(f"描述: {config['description']}")

        # 检查是否已经完成
        output_dir = Path(f"runs/{exp_name}")
        weights_file = output_dir / "weights" / "best.pt"

        if weights_file.exists():
            print(f"✅ 实验 {exp_name} 已完成，跳过训练")
            return True

        # 运行训练脚本
        try:
            # 这里可以调用具体的训练命令
            # 暂时使用基准训练脚本，实际应该根据配置调用不同的训练脚本
            cmd = [
                sys.executable, "scripts/train_baseline.py",
                "--experiment", exp_name,
                "--config", config['config_file']
            ]

            print(f"执行命令: {' '.join(cmd)}")

            # 在实际实现中，这里应该调用相应的训练函数
            # 为了演示，我们暂时使用基准训练
            if exp_name == 'baseline':
                from scripts.train_baseline import train_baseline
                result = train_baseline()
            else:
                print("⚠ 其他实验的训练功能开发中，暂时跳过")
                return False

            if result:
                config['status'] = 'completed'
                print(f"✅ 实验 {exp_name} 完成")
                return True
            else:
                config['status'] = 'failed'
                print(f"❌ 实验 {exp_name} 失败")
                return False

        except Exception as e:
            print(f"❌ 运行实验 {exp_name} 时出错: {e}")
            config['status'] = 'failed'
            return False

    def run_all_experiments(self, skip_completed=True):
        """运行所有实验"""
        print("🧪 开始运行消融实验")
        print("=" * 50)

        # 检查当前状态
        self.check_experiment_status()

        # 运行实验
        results = {}
        for exp_name in self.experiments.keys():
            if skip_completed and self.experiments[exp_name]['status'] == 'completed':
                print(f"\n⏭ 跳过已完成的实验: {exp_name}")
                results[exp_name] = 'skipped'
                continue

            success = self.run_single_experiment(exp_name)
            results[exp_name] = 'success' if success else 'failed'

            # 实验间暂停
            if exp_name != list(self.experiments.keys())[-1]:
                print("\n⏳ 等待5秒后开始下一个实验...")
                time.sleep(5)

        # 生成实验报告
        self.generate_experiment_report(results)

        return results

    def generate_experiment_report(self, results):
        """生成实验报告"""
        report_path = Path("results/ablation_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 消融实验报告\n\n")
            f.write("## 实验概览\n\n")
            f.write("| 实验名称 | 描述 | 状态 | 完成时间 |\n")
            f.write("|----------|------|------|----------|\n")

            for exp_name, result in results.items():
                config = self.experiments[exp_name]
                f.write(f"| {exp_name} | {config['description']} | {result} | - |\n")

            f.write("\n## 详细结果\n\n")
            # 这里可以添加更详细的结果分析

        print(f"📊 实验报告已生成: {report_path}")


def run_ablation_study():
    """运行消融实验"""
    runner = AblationRunner()

    # 设置实验配置
    runner.setup_experiments()

    # 运行所有实验
    results = runner.run_all_experiments(skip_completed=True)

    # 显示最终状态
    print("\n🎯 消融实验完成总结")
    print("=" * 50)
    for exp_name, result in results.items():
        status_icon = "✅" if result == 'success' else "❌" if result == 'failed' else "⏭"
        print(f"{status_icon} {exp_name}: {result}")

    return results


if __name__ == "__main__":
    run_ablation_study()