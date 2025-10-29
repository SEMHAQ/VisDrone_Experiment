#!/usr/bin/env python3
"""
训练集成EMA和BiFPN的YOLOv8模型
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO
from utils.path_manager import path_manager
from models.yolov8_ema_bifpn import create_yolov8_ema_bifpn_model


def train_ema_bifpn_model():
    """训练集成EMA和BiFPN的模型"""

    print("🚀 训练集成EMA和BiFPN的YOLOv8模型")
    print("=" * 50)

    # 验证环境
    if not path_manager.validate_paths():
        print("❌ 环境验证失败")
        return False

    # 创建模型
    print("🔄 创建模型...")
    model = create_yolov8_ema_bifpn_model(pretrained=True)

    # 训练配置
    print("⚙ 配置训练参数...")
    train_config = {
        'data': str(path_manager.dataset_config),
        'epochs': 80,
        'imgsz': 640,
        'batch': 16,
        'patience': 20,
        'device': 'cuda' if os.cuda.is_available() else 'cpu',
        'workers': 4,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'verbose': True,
        'project': str(path_manager.runs_dir),
        'name': 'yolov8_ema_bifpn'
    }

    # 开始训练
    print("🎯 开始训练...")
    try:
        results = model.train(**train_config)
        print("✅ 训练完成!")
        return results
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return False


if __name__ == "__main__":
    train_ema_bifpn_model()