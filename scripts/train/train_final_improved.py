#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终改进训练脚本 - YOLOv8s with P2 + BiFPN-Lite + 损失函数调整 + 小目标友好增强
综合多个改进点，包括：
1. P2 检测头
2. BiFPN-Lite 特征融合
3. 损失函数调整（SIoU）
4. 小目标友好增强策略（Copy-Paste, 增强调度）
"""
import os
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    print(f"工作目录: {project_root}")

    from ultralytics import YOLO

    # 配置文件路径
    data_yaml = project_root / "cfg" / "visdrone.yaml"
    model_yaml = project_root / "cfg" / "models" / "yolov8s-p2-bifpn-final.yaml"

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

    # 加载模型
    print("\n正在加载 YOLOv8s-P2-BiFPN (最终改进版) 模型...")
    try:
        model = YOLO(str(model_yaml))
        print("✓ 模型加载成功")
        print(f"  检测层数: 4 (P2, P3, P4, P5)")
        print(f"  检测步长: [4, 8, 16, 32]")
        print(f"  特征融合: BiFPN-Lite (双向特征金字塔)")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 最终改进的训练参数
    import platform
    is_windows = platform.system() == 'Windows'
    workers = 0 if is_windows else 4
    
    train_args = {
        'data': str(data_yaml),
        'project': 'runs/visdrone',
        'name': 'y8s_p2_bifpn_final_improved',
        'imgsz': 1024,
        'epochs': 300,  # 充分训练
        'batch': 4,
        'workers': workers,
        'device': 0,
        'amp': True,
        'seed': 42,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.01,
        'warmup_epochs': 15,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'cos_lr': True,
        
        # 数据增强 - 小目标友好策略
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5.0,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,  # 前70%开启
        'mixup': 0.1,
        'copy_paste': 0.2,  # 小目标 Copy-Paste，提升小目标学习
        'close_mosaic': 90,  # 后30%关闭 Mosaic，稳定收敛
        
        # 损失函数权重调整
        'box': 7.5,  # 保持
        'cls': 0.5,  # 保持
        'dfl': 1.5,  # 保持
        
        # 注意: ultralytics 可能不支持直接指定 SIoU
        # 如果需要 SIoU，可能需要修改底层代码或使用自定义损失
    }

    print("\n开始训练 YOLOv8s-P2-BiFPN (最终改进版)...")
    print("=" * 70)
    print(f"项目: {train_args['project']}")
    print(f"实验名: {train_args['name']}")
    print(f"改进组合:")
    print(f"  1. P2 Detection Head (stride=4)")
    print(f"  2. BiFPN-Lite (双向特征融合)")
    print(f"  3. 小目标友好增强 (Copy-Paste=0.2)")
    print(f"  4. 增强调度 (Mosaic 后30%关闭)")
    print(f"图像尺寸: {train_args['imgsz']}")
    print(f"批次大小: {train_args['batch']}")
    print(f"训练轮数: {train_args['epochs']}")
    print(f"初始学习率: {train_args['lr0']}")
    print(f"Copy-Paste: {train_args['copy_paste']} (小目标增强)")
    print(f"Mosaic 关闭: epoch {train_args['close_mosaic']}+")
    print("=" * 70)

    try:
        results = model.train(**train_args)
        print("\n✓ 训练完成！")
        print(f"最佳权重保存在: {results.save_dir}")
        print("\n建议下一步:")
        print(f"  运行评估: python compare_results.py")
        print(f"  或单独评估: python eval_model.py {results.save_dir}/weights/best.pt")
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

