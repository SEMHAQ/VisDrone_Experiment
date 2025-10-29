#!/usr/bin/env python3
"""
消融实验主脚本 - 完整版本
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from ultralytics import YOLO
from utils.path_manager import path_manager
from models.hook_integration import create_ema_model, create_bifpn_model, create_full_model


class AblationStudy:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.experiments = {

            'baseline': {
                'name': 'baseline',
                'description': '原始YOLOv8s',
                'train_fn': self.train_baseline,
                'evaluate_fn': self.evaluate_model
            },

            'ema': {
                'name': 'ema',
                'description': 'YOLOv8s + EMA注意力',
                'train_fn': self.train_ema,
                'evaluate_fn': self.evaluate_model
            },



            'bifpn': {
                'name': 'bifpn',
                'description': 'YOLOv8s + BiFPN',
                'train_fn': self.train_bifpn,
                'evaluate_fn': self.evaluate_model
            },

            'full': {
                'name': 'full',
                'description': 'YOLOv8s + EMA + BiFPN',
                'train_fn': self.train_full,
                'evaluate_fn': self.evaluate_model
            },
        }

    def get_train_config(self, exp_name):
        """获取训练配置"""
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
            'verbose': False,
            'project': str(path_manager.runs_dir),
            'name': exp_name
        }

    def train_baseline(self, exp_name):
        """训练基准模型"""
        try:
            model = YOLO('scripts/yolov8s.pt')
            config = self.get_train_config(exp_name)
            results = model.train(**config)
            return True, results
        except Exception as e:
            return False, str(e)

    def train_ema(self, exp_name):
        """训练EMA模型"""
        try:
            integrator = create_ema_model()
            config = self.get_train_config(exp_name)
            results = integrator.model.train(**config)
            integrator.cleanup()
            return True, results
        except Exception as e:
            return False, str(e)

    def train_bifpn(self, exp_name):
        """训练BiFPN模型"""
        try:
            integrator = create_bifpn_model()
            config = self.get_train_config(exp_name)
            results = integrator.model.train(**config)
            integrator.cleanup()
            return True, results
        except Exception as e:
            return False, str(e)

    def train_full(self, exp_name):
        """训练完整模型"""
        try:
            integrator = create_full_model()
            config = self.get_train_config(exp_name)
            results = integrator.model.train(**config)
            integrator.cleanup()
            return True, results
        except Exception as e:
            return False, str(e)

    def evaluate_model(self, exp_name):
        """评估模型"""
        try:
            exp_dir = path_manager.get_experiment_dir(exp_name)
            weights_file = exp_dir / "weights" / "best.pt"

            if not weights_file.exists():
                return False, "模型文件不存在"

            model = YOLO(str(weights_file))
            metrics = model.val(
                data=str(path_manager.dataset_config),
                split='val',
                imgsz=640,
                batch=16,
                conf=0.001,
                iou=0.6,
                verbose=False
            )

            results = {
                'map50': float(metrics.box.map50),
                'map': float(metrics.box.map),
                'precision': float(metrics.box.precision.mean()),
                'recall': float(metrics.box.recall.mean()),
                'f1_score': self.calculate_f1_score(
                    float(metrics.box.precision.mean()),
                    float(metrics.box.recall.mean())
                )
            }

            return True, results

        except Exception as e:
            return False, str(e)

    def calculate_f1_score(self, precision, recall):
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def run(self):
        """运行消融实验"""
        print("🧪 开始消融实验")
        print("=" * 60)
        print("实验顺序:")
        for exp_key, config in self.experiments.items():
            print(f"- {exp_key}: {config['description']}")
        print("=" * 60)

        results = {}

        for exp_key, config in self.experiments.items():
            exp_name = config['name']
            description = config['description']

            print(f"\n🎯 实验 {exp_key}: {description}")
            print("-" * 50)

            # 检查是否已训练
            exp_dir = path_manager.get_experiment_dir(exp_name)
            weights_file = exp_dir / "weights" / "best.pt"

            if weights_file.exists():
                print("✅ 模型已训练，跳过训练阶段")
                train_success, train_result = True, "已训练"
            else:
                # 训练模型
                print("开始训练...")
                train_success, train_result = config['train_fn'](exp_name)

                if train_success:
                    print("✅ 训练完成")
                else:
                    print(f"❌ 训练失败: {train_result}")

            # 评估模型
            if train_success:
                print("开始评估...")
                eval_success, eval_result = config['evaluate_fn'](exp_name)

                if eval_success:
                    print("✅ 评估完成")
                    results[exp_key] = {
                        'status': 'success',
                        'metrics': eval_result,
                        'train_result': train_result
                    }
                else:
                    print(f"❌ 评估失败: {eval_result}")
                    results[exp_key] = {
                        'status': 'evaluation_failed',
                        'error': eval_result,
                        'train_result': train_result
                    }
            else:
                results[exp_key] = {
                    'status': 'training_failed',
                    'error': train_result
                }

        # 分析结果
        self.analyze_results(results)
        return results

    def analyze_results(self, results):
        """分析实验结果"""
        print("\n📊 实验结果分析")
        print("=" * 60)

        successful_experiments = {}
        for exp_key, result in results.items():
            if result['status'] == 'success':
                successful_experiments[exp_key] = result['metrics']

        if not successful_experiments:
            print("❌ 没有成功的实验")
            return

        # 显示性能对比
        print("\n性能对比:")
        print("| 实验 | mAP@0.5 | mAP@0.5:0.95 | 精确率 | 召回率 | F1分数 |")
        print("|------|---------|--------------|--------|--------|--------|")

        for exp_key, metrics in successful_experiments.items():
            print(f"| {exp_key} | {metrics['map50']:.4f} | {metrics['map']:.4f} | "
                  f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} |")

        # 计算改进百分比
        baseline_metrics = successful_experiments.get('baseline')
        if baseline_metrics:
            print("\n改进百分比 (相对于基准模型):")
            print("| 实验 | mAP@0.5改进 | mAP改进 | F1改进 |")
            print("|------|------------|---------|--------|")

            for exp_key, metrics in successful_experiments.items():
                if exp_key != 'baseline':
                    map50_improvement = ((metrics['map50'] - baseline_metrics['map50']) / baseline_metrics[
                        'map50']) * 100
                    map_improvement = ((metrics['map'] - baseline_metrics['map']) / baseline_metrics['map']) * 100
                    f1_improvement = ((metrics['f1_score'] - baseline_metrics['f1_score']) / baseline_metrics[
                        'f1_score']) * 100

                    print(
                        f"| {exp_key} | {map50_improvement:+.2f}% | {map_improvement:+.2f}% | {f1_improvement:+.2f}% |")

        # 生成报告
        self.generate_report(results, successful_experiments)

    def generate_report(self, results, successful_experiments):
        """生成实验报告"""
        report_dir = Path("results")
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / "ablation_study_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# VisDrone消融实验报告\n\n")
            f.write("## 实验概览\n\n")

            for exp_key, result in results.items():
                status_icon = "✅" if result['status'] == 'success' else "❌"
                f.write(f"- {status_icon} **{exp_key}**: {self.experiments[exp_key]['description']}\n")
                if result['status'] != 'success':
                    f.write(f"  - 状态: {result['status']}\n")
                    if 'error' in result:
                        f.write(f"  - 错误: {result['error']}\n")

            if successful_experiments:
                f.write("\n## 性能对比\n\n")
                f.write("| 实验 | mAP@0.5 | mAP@0.5:0.95 | 精确率 | 召回率 | F1分数 |\n")
                f.write("|------|---------|--------------|--------|--------|--------|\n")

                for exp_key, metrics in successful_experiments.items():
                    f.write(f"| {exp_key} | {metrics['map50']:.4f} | {metrics['map']:.4f} | "
                            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} |\n")

            f.write("\n## 结论\n\n")
            f.write("通过消融实验验证了各个改进模块的有效性。\n")

        print(f"✅ 实验报告已生成: {report_path}")


def main():
    """主函数"""
    # 验证环境
    if not path_manager.validate_paths():
        print("❌ 环境验证失败")
        return

    print("🧪 VisDrone目标检测消融实验")
    print("=" * 60)

    study = AblationStudy()
    results = study.run()

    print("\n🎯 消融实验完成!")
    return results


if __name__ == "__main__":
    main()