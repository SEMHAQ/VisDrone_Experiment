#!/usr/bin/env python3
"""
BiFPN独立实验
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO
from utils.path_manager import path_manager


class BiFPNOnlyExperiment:
    """BiFPN独立实验"""

    def __init__(self):
        self.exp_name = "bifpn_only"
        self.description = "YOLOv8s + BiFPN特征金字塔"

    def run(self):
        """运行BiFPN独立实验"""
        print("=" * 60)
        print("       BiFPN独立实验")
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

        # 应用BiFPN改进
        print("🔧 应用BiFPN改进...")
        model = self.apply_bifpn_improvements(model)

        # 训练配置
        train_config = {
            'data': str(path_manager.dataset_config),
            'epochs': 80,
            'imgsz': 640,
            'batch': 16,
            'patience': 20,
            'device': 'cuda' if os.cuda.is_available() else 'cpu',
            'workers': 4,
            'project': str(path_manager.runs_dir),
            'name': self.exp_name,
            'exist_ok': True,
            'verbose': True
        }

        # 训练模型
        print("🚀 开始训练BiFPN模型...")
        try:
            results = model.train(**train_config)
            print("✅ BiFPN模型训练完成")
            return True
        except Exception as e:
            print(f"❌ BiFPN模型训练失败: {e}")
            return False

    def apply_bifpn_improvements(self, model):
        """应用BiFPN改进 - 需要具体实现"""
        print("⚠ BiFPN集成功能开发中，暂时使用基础模型")
        # 这里需要实现BiFPN的实际集成
        return model

    def evaluate(self):
        """评估BiFPN模型"""
        print(f"\n📊 评估BiFPN模型...")

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

            print(f"✅ BiFPN模型评估完成:")
            print(f"   mAP@0.5: {result['map50']:.4f}")
            print(f"   mAP: {result['map']:.4f}")
            print(f"   精确率: {result['precision']:.4f}")
            print(f"   召回率: {result['recall']:.4f}")
            print(f"   F1分数: {result['f1_score']:.4f}")

            return result

        except Exception as e:
            print(f"❌ 评估失败: {e}")
            return None

    def calculate_f1_score(self, precision, recall):
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


def main():
    """主函数"""
    experiment = BiFPNOnlyExperiment()

    # 运行实验
    success = experiment.run()

    if success:
        # 评估模型
        experiment.evaluate()
        print("\n🎯 BiFPN独立实验完成!")
    else:
        print("\n❌ BiFPN独立实验失败")


if __name__ == "__main__":
    main()