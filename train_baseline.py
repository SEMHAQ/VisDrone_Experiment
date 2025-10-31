#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线训练脚本 - YOLOv8s on VisDrone
适配 12GB RTX 3060
"""
import os
import sys
from pathlib import Path

def main():
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

    # 检查数据完整性
    print("\n检查数据完整性...")
    base_path = project_root / "VisDrone2YOLO"
    for split in ['train', 'val']:
        images_dir = base_path / f"VisDrone2019-DET-{split}" / "images"
        labels_dir = base_path / f"VisDrone2019-DET-{split}" / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"✗ {split} 数据集目录不存在")
            sys.exit(1)
        
        # 检查图片文件
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        label_files = list(labels_dir.glob("*.txt"))
        
        if len(image_files) == 0:
            print(f"\n✗ 错误: {split} 的 images 目录为空！")
            print(f"   请先下载 VisDrone 数据集的图片文件")
            print(f"   图片应该放在: {images_dir}")
            print(f"\n   下载地址: https://github.com/VisDrone/VisDrone-Dataset")
            print(f"   需要下载: VisDrone2019-DET-{split}.zip")
            sys.exit(1)
        
        if len(label_files) == 0:
            print(f"✗ {split} 的 labels 目录为空")
            sys.exit(1)
        
        print(f"✓ {split}: {len(image_files)} 张图片, {len(label_files)} 个标签")

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
    # Windows 上 workers 设置为 0 避免多进程问题，或使用较小值
    import platform
    is_windows = platform.system() == 'Windows'
    workers = 0 if is_windows else 4  # Windows 单进程，Linux/Mac 可以多进程
    
    train_args = {
        'data': str(data_yaml),
        'project': 'runs/visdrone',
        'name': 'baseline_y8s_1024_adamw',
        'imgsz': 1024,
        'epochs': 300,
        'batch': 8,
        'workers': workers,
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
    print(f"工作进程数: {train_args['workers']} ({'单进程模式' if is_windows else '多进程模式'})")
    print(f"总轮数: {train_args['epochs']}")
    print("=" * 60)

    try:
        results = model.train(**train_args)
        print("\n✓ 训练完成！")
        print(f"最佳权重保存在: {results.save_dir}")
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

