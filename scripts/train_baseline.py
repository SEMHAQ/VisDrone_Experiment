#!/usr/bin/env python3
"""
基准模型训练脚本
"""

import sys
import os

from torch.xpu import device

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from ultralytics import YOLO
from utils.path_manager import path_manager


def train_baseline():
    """训练基准模型"""

    print("🎯 开始训练基准模型 (YOLOv8s)")
    print("=" * 50)

    # 验证环境
    if not path_manager.validate_paths():
        print("❌ 环境验证失败，请先运行 verify_environment.py")
        return

    # 检查配置文件
    if not path_manager.dataset_config.exists():
        print(f"❌ 配置文件不存在: {path_manager.dataset_config}")
        return

    print(f"📁 数据集路径: {path_manager.dataset_root}")
    print(f"⚙ 配置文件: {path_manager.dataset_config}")

    # 创建输出目录
    output_dir = path_manager.get_experiment_dir("baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print("🔄 加载YOLOv8s模型...")
    model = YOLO('yolov8s.pt')

    # 训练配置
    print("🚀 开始训练...")
    results = model.train(
        data=str(path_manager.dataset_config),
        epochs=80,
        imgsz=640,
        batch=16,
        patience=20,
        save=True,
        exist_ok=True,
        device='cpu',  # 使用GPU，如果是CPU改为 'cpu'
        workers=4,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        verbose=True,
        project=str(path_manager.runs_dir),
        name='baseline'
    )

    print("✅ 基准模型训练完成!")
    print(f"📊 结果保存在: {output_dir}")

    return results


if __name__ == "__main__":
    train_baseline()