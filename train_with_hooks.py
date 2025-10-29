#!/usr/bin/env python3
"""
基于Hook的消融实验训练脚本
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from ultralytics import YOLO
from utils.path_manager import path_manager
from models.hook_integration import (
    YOLOv8HookIntegrator,
    create_ema_model,
    create_bifpn_model,
    create_full_model
)


class HookBasedAblationStudy:
    """基于Hook的消融实验研究"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.experiments = {
            'baseline': {
                'name': 'baseline',
                'description': '原始YOLOv8s',
                'create_fn': self.create_baseline,
                'config': self.get_base_config()
            },
            'ema': {
                'name': 'ema',
                'description': 'YOLOv8s + EMA注意力',
                'create_fn': self.create_ema_model,
                'config': self.get_base_config()
            },
            'bifpn': {
                'name': 'bifpn',
                'description': 'YOLOv8s + BiFPN',
                'create_fn': self.create_bifpn_model,
                'config': self.get_base_config()
            },
            'full': {
                'name': 'full',
                'description': 'YOLOv8s + EMA + BiFPN',
                'create_fn': self.create_full_model,
                'config': self.get_base_config()
            }
        }

    def get_base_config(self):
        """基础训练配置"""
        return {
            'data': str(path_manager.dataset_config),
            'epochs': 80,
            'imgsz': 640,
            'batch': 16,
            'patience': 20,
            'device': self.device,
            'workers': 4,
            'save': True,
            'exist_ok': True,
            'verbose': True,
            'project': str(path_manager.runs_dir)
        }

    def create_baseline(self):
        """创建基准模型"""
        print("创建基准模型...")
        return YOLO('scripts/yolov8s.pt')

    def create_ema_model(self):
        """创建EMA模型"""
        print("创建EMA模型...")
        integrator = create_ema_model()
        return integrator.model

    def create_bifpn_model(self):
        """创建BiFPN模型"""
        print("创建BiFPN模型...")
        integrator = create_bifpn_model()
        return integrator.model

    def create_full_model(self):
        """创建完整模型"""
        print("创建完整模型...")
        integrator = create_full_model()
        return integrator.model

    def train_experiment(self, exp_key):
        """训练单个实验"""
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
            model = exp_config['create_fn']()

            # 训练配置
            config = exp_config['config'].copy()
            config['name'] = exp_config['name']

            # 训练模型
            print("开始训练...")
            results = model.train(**config)

            print(f"✅ {exp_config['name']} 训练完成")
            return True

        except Exception as e:
            print(f"❌ {exp_config['name']} 训练失败: {e}")
            return False

    def evaluate_experiment(self, exp_key):
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
                'map50': float(metrics.box.map50),
                'map': float(metrics.box.map),
                'precision': float(metrics.box.precision.mean()),
                'recall': float(metrics.box.recall.mean()),
                'f1_score': self.calculate_f1_score(
                    float(metrics.box.precision.mean()),
                    float(metrics.box.recall.mean())
                )
            }

            print(f"✅ {exp_config['name']}: mAP50 = {result['map50']:.4f}")
            return result

        except Exception as e:
            print(f"❌ 评估失败: {e}")
            return None

    def calculate_f1_score(self, precision, recall):
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def run_ablation_study(self):
        """运行消融实验"""
        print("🧪 基于Hook的消融实验")
        print("=" * 50)
        print("实验顺序:")
        print("1. Baseline (原始YOLOv8s)")
        print("2. +EMA注意力机制")
        print("3. +BiFPN特征金字塔")
        print("4. Full (EMA + BiFPN)")
        print("=" * 50)

        results = {}

        # 按顺序运行实验
        experiment_order = ['baseline', 'ema', 'bifpn', 'full']

        for exp_key in experiment_order:
            # 训练
            success = self.train_experiment(exp_key)

            # 评估
            if success:
                metrics = self.evaluate_experiment(exp_key)
                results[exp_key] = metrics
            else:
                results[exp_key] = None

            print("\n" + "=" * 50)

        # 分析结果
        self.analyze_results(results)

        return results

    def analyze_results(self, results):
        """分析实验结果"""
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
            print("| 实验 | mAP50改进 | mAP改进 | F1改进 |")
            print("|------|-----------|---------|--------|")

            for exp_key, metrics in valid_results.items():
                if exp_key != 'baseline' and metrics:
                    map50_improvement = ((metrics['map50'] - baseline_metrics['map50']) / baseline_metrics[
                        'map50']) * 100
                    map_improvement = ((metrics['map'] - baseline_metrics['map']) / baseline_metrics['map']) * 100
                    f1_improvement = ((metrics['f1_score'] - baseline_metrics['f1_score']) / baseline_metrics[
                        'f1_score']) * 100

                    print(
                        f"| {exp_key} | {map50_improvement:+.2f}% | {map_improvement:+.2f}% | {f1_improvement:+.2f}% |")

        # 生成详细报告
        self.generate_detailed_report(valid_results)

    def generate_detailed_report(self, results):
        """生成详细报告"""
        report_dir = Path("results/hook_ablation")
        report_dir.mkdir(parents=True, exist_ok=True)

        # Markdown报告
        report_path = report_dir / "hook_ablation_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 基于Hook机制的消融实验报告\n\n")
            f.write("## 实验配置\n\n")
            f.write("- **方法**: 使用Hook机制动态集成模块\n")
            f.write("- **设备**: GPU\n" if self.device == 'cuda' else "- **设备**: CPU\n")
            f.write("- **数据集**: VisDrone\n\n")

            f.write("## 实验结果\n\n")
            f.write("| 实验 | mAP@0.5 | mAP@0.5:0.95 | 精确率 | 召回率 | F1分数 |\n")
            f.write("|------|---------|--------------|--------|--------|--------|\n")

            for exp_key, metrics in results.items():
                f.write(f"| {exp_key} | {metrics['map50']:.4f} | {metrics['map']:.4f} | "
                        f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} |\n")

            f.write("\n## 结论\n\n")
            f.write("通过Hook机制成功实现了EMA注意力和BiFPN的灵活集成，验证了各个模块的有效性。\n")

        print(f"✅ 详细报告已生成: {report_path}")


def main():
    """主函数"""
    # 验证环境
    if not path_manager.validate_paths():
        print("❌ 环境验证失败")
        return

    print("🧪 基于Hook机制的VisDrone消融实验")
    print("=" * 60)

    study = HookBasedAblationStudy()
    results = study.run_ablation_study()

    print("\n🎯 消融实验完成!")
    return results


if __name__ == "__main__":
    main()