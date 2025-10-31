#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线训练脚本 - YOLOv8s on VisDrone
适配 12GB RTX 3060
"""
import os
import sys
from pathlib import Path

# 确保在项目根目录
project_root = Path(__file__).parent.absolute()
os.chdir(project_root)
print(f"工作目录: {project_root}")

from ultralytics import YOLO

# 配置文件路径
data_yaml = project_root / "cfg" / "visdrone.yaml"
train_cfg = project_root / "cfg" / "train_base1024.yaml"

# 检查配置文件是否存在
if not data_yaml.exists():
    print(f"错误: 数据配置文件不存在: {data_yaml}")
    sys.exit(1)

if not train_cfg.exists():
    print(f"错误: 训练配置文件不存在: {train_cfg}")
    sys.exit(1)

print(f"数据配置: {data_yaml}")
print(f"训练配置: {train_cfg}")

# 加载模型
print("\n正在加载 YOLOv8s 模型...")
try:
    model = YOLO('yolov8s.pt')
    print("✓ 模型加载成功")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 训练参数
train_args = {
    'data': str(data_yaml),
    'project': 'runs/visdrone',
    'name': 'baseline_y8s_1024_adamw',
    'imgsz': 1024,
    'epochs': 300,
    'batch': 8,
    'workers': 4,
    'device': 0,
    'amp': True,
    'seed': 42,
    'optimizer': 'AdamW',
    'lr0': 0.0025,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.01,
    'warmup_epochs': 10,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'cos_lr': True,
    'ema': True,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 5.0,
    'translate': 0.1,
    'scale': 0.5,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.1,
    'copy_paste': 0.0,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
}

print("\n开始训练...")
print("=" * 60)
print(f"项目: {train_args['project']}")
print(f"实验名: {train_args['name']}")
print(f"图像尺寸: {train_args['imgsz']}")
print(f"批次大小: {train_args['batch']}")
print(f"总轮数: {train_args['epochs']}")
print("=" * 60)

try:
    results = model.train(**train_args)
    print("\n✓ 训练完成！")
except Exception as e:
    print(f"\n✗ 训练失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

