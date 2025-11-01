#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本 - YOLOv8s with P2 Detection Head
添加 stride=4 的 P2 检测头以提升小目标检测能力
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
    model_yaml = project_root / "cfg" / "models" / "yolov8s-p2.yaml"

    # 检查配置文件是否存在
    if not data_yaml.exists():
        print(f"错误: 数据配置文件不存在: {data_yaml}")
        sys.exit(1)

    if not model_yaml.exists():
        print(f"错误: 模型配置文件不存在: {model_yaml}")
        sys.exit(1)

    print(f"数据配置: {data_yaml}")
    print(f"模型配置: {model_yaml}")

    # 检查数据完整性
    print("\n检查数据完整性...")
    base_path = project_root / "VisDrone2YOLO"
    for split in ['train', 'val']:
        images_dir = base_path / f"VisDrone2019-DET-{split}" / "images"
        labels_dir = base_path / f"VisDrone2019-DET-{split}" / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"✗ {split} 数据集目录不存在")
            sys.exit(1)
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        label_files = list(labels_dir.glob("*.txt"))
        
        if len(image_files) == 0:
            print(f"\n✗ 错误: {split} 的 images 目录为空！")
            sys.exit(1)
        
        if len(label_files) == 0:
            print(f"✗ {split} 的 labels 目录为空")
            sys.exit(1)
        
        print(f"✓ {split}: {len(image_files)} 张图片, {len(label_files)} 个标签")

    # 加载模型（从配置文件创建）
    print("\n正在加载 YOLOv8s-P2 模型...")
    try:
        model = YOLO(str(model_yaml))
        print("✓ 模型加载成功")
        print(f"  检测层数: 4 (P2, P3, P4, P5)")
        print(f"  检测步长: [4, 8, 16, 32]")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 训练参数（与基线保持一致，便于对比）
    import platform
    is_windows = platform.system() == 'Windows'
    workers = 0 if is_windows else 4
    
    train_args = {
        'data': str(data_yaml),
        'project': 'runs/visdrone',
        'name': 'y8s_p2_1024_adamw',
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

    print("\n开始训练 YOLOv8s-P2...")
    print("=" * 60)
    print(f"项目: {train_args['project']}")
    print(f"实验名: {train_args['name']}")
    print(f"改进: +P2 Detection Head (stride=4)")
    print(f"图像尺寸: {train_args['imgsz']}")
    print(f"批次大小: {train_args['batch']}")
    print(f"工作进程数: {train_args['workers']} ({'单进程模式' if is_windows else '多进程模式'})")
    print(f"总轮数: {train_args['epochs']}")
    print("=" * 60)

    try:
        results = model.train(**train_args)
        print("\n✓ 训练完成！")
        print(f"最佳权重保存在: {results.save_dir}")
        print("\n建议下一步:")
        print(f"  运行评估: python eval_model.py {results.save_dir}/weights/best.pt")
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

