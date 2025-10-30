#!/usr/bin/env python3
"""
验证所有实验结果的脚本
"""

import json
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO
from utils.path_manager import path_manager


class ExperimentVerifier:
    """实验验证器"""

    def __init__(self):
        self.experiments = [
            'baseline',
            'ema_only',
            'bifpn_only',
            'full_model'
        ]

        self.results = {}

    def verify_all_experiments(self):
        """验证所有实验"""
        print("=" * 60)
        print("       实验验证")
        print("=" * 60)

        for exp_name in self.experiments:
            print(f"\n🔍 验证实验: {exp_name}")
            print("-" * 50)

            # 检查模型文件
            exp_dir = path_manager.get_experiment_dir(exp_name)
            weights_file = exp_dir / "weights" / "best.pt"

            if not weights_file.exists():
                print(f"❌ 模型文件不存在: {weights_file}")
                self.results[exp_name] = {'status': 'missing', 'error': '模型文件不存在'}
                continue

            # 评估模型
            metrics = self.evaluate_model(exp_name, weights_file)

            if metrics:
                self.results[exp_name] = {
                    'status': 'completed',
                    'metrics': metrics
                }
                print(f"✅ {exp_name} 验证完成")
            else:
                self.results[exp_name] = {
                    'status': 'failed',
                    'error': '评估失败'
                }
                print(f"❌ {exp_name} 验证失败")

        # 生成报告
        self.generate_report()

        return self.results

    def evaluate_model(self, exp_name, weights_path):
        """评估单个模型"""
        try:
            model = YOLO(str(weights_path))

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

            return {
                'map50': float(metrics.box.map50),
                'map': float(metrics.box.map),
                'precision': float(metrics.box.p.mean()),
                'recall': float(metrics.box.r.mean()),
                'f1_score': self.calculate_f1_score(
                    float(metrics.box.p.mean()),
                    float(metrics.box.r.mean())
                )
            }

        except Exception as e:
            print(f"❌ 评估失败: {e}")
            return None

    def calculate_f1_score(self, precision, recall):
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def generate_report(self):
        """生成验证报告"""
        print("\n📊 验证结果报告")
        print("=" * 60)

        # 创建结果目录
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 生成性能对比表
        completed_exps = {k: v for k, v in self.results.items() if v['status'] == 'completed'}

        if completed_exps:
            df_data = []
            for exp_name, result in completed_exps.items():
                metrics = result['metrics']
                df_data.append({
                    '实验': exp_name,
                    'mAP@0.5': metrics['map50'],
                    'mAP@0.5:0.95': metrics['map'],
                    '精确率': metrics['precision'],
                    '召回率': metrics['recall'],
                    'F1分数': metrics['f1_score']
                })

            df = pd.DataFrame(df_data)
            print("\n📈 性能对比:")
            print(df.to_string(index=False, float_format='%.4f'))

            # 保存CSV
            csv_path = results_dir / "performance_comparison.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ 性能对比表已保存: {csv_path}")

        # 保存详细结果
        json_path = results_dir / "verification_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✅ 详细结果已保存: {json_path}")

        # 状态总结
        print(f"\n📋 实验状态总结:")
        for exp_name, result in self.results.items():
            status_icon = "✅" if result['status'] == 'completed' else "❌"
            print(f"{status_icon} {exp_name}: {result['status']}")


def main():
    """主函数"""
    verifier = ExperimentVerifier()
    results = verifier.verify_all_experiments()

    print("\n🎯 验证完成!")
    return results


if __name__ == "__main__":
    main()