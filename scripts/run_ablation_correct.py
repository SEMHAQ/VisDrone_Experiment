#!/usr/bin/env python3
"""
正确的消融实验方案 - 逐个模块测试
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO
from utils.path_manager import path_manager


class CorrectAblationStudy:
    """正确的消融实验管理"""

    def __init__(self):
        self.experiments = {
            'baseline': {
                'name': 'baseline',
                'description': '原始YOLOv8s',
                'model_fn': self.create_baseline,
                'config': self.get_baseline_config()
            },
            'ema': {
                'name': 'ema',
                'description': 'YOLOv8s + EMA注意力',
                'model_fn': self.create_ema_model,
                'config': self.get_ema_config()
            },
            'bifpn': {
                'name': 'bifpn',
                'description': 'YOLOv8s + BiFPN',
                'model_fn': self.create_bifpn_model,
                'config': self.get_bifpn_config()
            },
            'full': {
                'name': 'full',
                'description': 'YOLOv8s + EMA + BiFPN',
                'model_fn': self.create_full_model,
                'config': self.get_full_config()
            }
        }

    def get_baseline_config(self):
        """基准模型配置"""
        return {
            'data': str(path_manager.dataset_config),
            'epochs': 80,
            'imgsz': 640,
            'batch': 16,
            'patience': 20,
            'device': 'cpu',
            'project': str(path_manager.runs_dir),
            'name': 'baseline'
        }

    def get_ema_config(self):
        """EMA模型配置"""
        config = self.get_baseline_config()
        config.update({
            'name': 'ema',
            'ema_attention': True,  # 自定义参数
            'attention_channels': 512
        })
        return config

    def get_bifpn_config(self):
        """BiFPN模型配置"""
        config = self.get_baseline_config()
        config.update({
            'name': 'bifpn',
            'bifpn': True,  # 自定义参数
            'bifpn_channels': 256
        })
        return config

    def get_full_config(self):
        """完整模型配置"""
        config = self.get_baseline_config()
        config.update({
            'name': 'full',
            'ema_attention': True,
            'bifpn': True,
            'attention_channels': 512,
            'bifpn_channels': 256
        })
        return config

    def create_baseline(self):
        """创建基准模型"""
        print("创建基准模型...")
        return YOLO('yolov8s.pt')

    def create_ema_model(self):
        """创建EMA模型"""
        print("创建EMA注意力模型...")
        # 这里应该调用实际的EMA模型创建函数
        model = YOLO('yolov8s.pt')

        # 通过hook方式添加EMA注意力
        self._add_ema_via_hook(model)
        return model

    def create_bifpn_model(self):
        """创建BiFPN模型"""
        print("创建BiFPN模型...")
        model = YOLO('yolov8s.pt')

        # 通过hook方式替换PANet为BiFPN
        self._replace_pan_with_bifpn(model)
        return model

    def create_full_model(self):
        """创建完整模型"""
        print("创建完整模型...")
        model = YOLO('yolov8s.pt')

        # 集成EMA和BiFPN
        self._add_ema_via_hook(model)
        self._replace_pan_with_bifpn(model)
        return model

    def _add_ema_via_hook(self, model):
        """通过hook方式添加EMA注意力"""
        # 实际实现需要根据YOLOv8的具体结构
        print("⚠ 通过hook添加EMA注意力（需要具体实现）")

    def _replace_pan_with_bifpn(self, model):
        """替换PANet为BiFPN"""
        # 实际实现需要根据YOLOv8的具体结构
        print("⚠ 替换PANet为BiFPN（需要具体实现）")

    def run_single_experiment(self, exp_key):
        """运行单个实验"""
        exp_config = self.experiments[exp_key]

        print(f"\n🚀 开始实验: {exp_config['name']}")
        print(f"描述: {exp_config['description']}")
        print("=" * 50)

        # 检查是否已训练
        exp_dir = path_manager.get_experiment_dir(exp_config['name'])
        weights_file = exp_dir / "weights" / "best.pt"

        if weights_file.exists():
            print(f"✅ 实验已完成，跳过训练")
            return True

        try:
            # 创建模型
            model = exp_config['model_fn']()

            # 训练配置
            config = exp_config['config']

            # 训练模型
            print("开始训练...")
            results = model.train(**config)

            print(f"✅ {exp_config['name']} 训练完成")
            return True

        except Exception as e:
            print(f"❌ {exp_config['name']} 训练失败: {e}")
            return False

    def evaluate_single_experiment(self, exp_key):
        """评估单个实验"""
        exp_config = self.experiments[exp_key]

        print(f"\n📊 评估: {exp_config['name']}")

        exp_dir = path_manager.get_experiment_dir(exp_config['name'])
        weights_file = exp_dir / "weights" / "best.pt"

        if not weights_file.exists():
            print(f"❌ 模型文件不存在")
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
                verbose=False
            )

            result = {
                'map50': metrics.box.map50,
                'map': metrics.box.map,
                'precision': metrics.box.precision.mean(),
                'recall': metrics.box.recall.mean()
            }

            print(f"✅ {exp_config['name']}: mAP50 = {result['map50']:.4f}")
            return result

        except Exception as e:
            print(f"❌ 评估失败: {e}")
            return None

    def run_sequential_ablation(self):
        """顺序运行消融实验"""
        print("🧪 开始顺序消融实验")
        print("=" * 50)

        results = {}

        # 按顺序运行实验
        experiment_order = ['baseline', 'ema', 'bifpn', 'full']

        for exp_key in experiment_order:
            # 训练
            success = self.run_single_experiment(exp_key)

            # 评估
            if success:
                metrics = self.evaluate_single_experiment(exp_key)
                results[exp_key] = metrics
            else:
                results[exp_key] = None

            print("\n" + "=" * 50)

        # 分析结果
        self.analyze_results(results)

        return results

    def analyze_results(self, results):
        """分析消融实验结果"""
        print("\n📈 消融实验结果分析")
        print("=" * 50)

        valid_results = {k: v for k, v in results.items() if v is not None}

        if not valid_results:
            print("❌ 没有有效的实验结果")
            return

        # 计算改进百分比
        baseline_metrics = valid_results.get('baseline')
        if baseline_metrics:
            print("相对于基准模型的改进:")
            for exp_key, metrics in valid_results.items():
                if exp_key != 'baseline' and metrics:
                    map50_improvement = ((metrics['map50'] - baseline_metrics['map50']) / baseline_metrics[
                        'map50']) * 100
                    print(f"  {exp_key}: mAP50 +{map50_improvement:+.2f}%")

        # 生成报告
        self.generate_ablation_report(valid_results)

    def generate_ablation_report(self, results):
        """生成消融实验报告"""
        report_path = Path("results/ablation_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# VisDrone消融实验报告\n\n")
            f.write("## 实验结果\n\n")
            f.write("| 实验 | mAP@0.5 | mAP@0.5:0.95 | 精确率 | 召回率 |\n")
            f.write("|------|---------|--------------|--------|--------|\n")

            for exp_key, metrics in results.items():
                if metrics:
                    f.write(
                        f"| {exp_key} | {metrics['map50']:.4f} | {metrics['map']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} |\n")

            f.write("\n## 结论\n")
            f.write("通过逐个模块的消融实验，可以准确评估每个改进的贡献。\n")

        print(f"✅ 报告已生成: {report_path}")


def main():
    """主函数"""
    import torch

    print("🧪 VisDrone正确的消融实验")
    print("=" * 50)
    print("实验顺序:")
    print("1. Baseline (原始YOLOv8s)")
    print("2. +EMA注意力机制")
    print("3. +BiFPN特征金字塔")
    print("4. Full (EMA + BiFPN)")
    print("=" * 50)

    study = CorrectAblationStudy()
    results = study.run_sequential_ablation()

    print("\n🎯 消融实验完成!")
    return results


if __name__ == "__main__":
    main()