#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 VisDrone 数据完整性
验证图片和标签文件是否匹配
"""
import os
from pathlib import Path

def check_dataset_split(split_name, base_path):
    """检查某个数据集分割（train/val/test）的完整性"""
    split_path = Path(base_path) / f"VisDrone2019-DET-{split_name}"
    images_dir = split_path / "images"
    labels_dir = split_path / "labels"
    annotations_dir = split_path / "annotations"
    
    print(f"\n{'='*60}")
    print(f"检查 {split_name} 数据集")
    print(f"{'='*60}")
    
    # 检查目录是否存在
    if not images_dir.exists():
        print(f"✗ images 目录不存在: {images_dir}")
        return False
    if not labels_dir.exists():
        print(f"✗ labels 目录不存在: {labels_dir}")
        return False
    
    # 获取图片文件（支持 .jpg, .png）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
        image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
    
    # 获取标签文件
    label_files = list(labels_dir.glob("*.txt"))
    
    print(f"✓ images 目录: {images_dir}")
    print(f"  - 找到图片文件: {len(image_files)} 个")
    print(f"✓ labels 目录: {labels_dir}")
    print(f"  - 找到标签文件: {len(label_files)} 个")
    
    if len(image_files) == 0:
        print(f"\n⚠ 警告: {split_name} 的 images 目录为空！")
        print(f"  请下载 VisDrone 数据集的图片文件并放入: {images_dir}")
        return False
    
    if len(label_files) == 0:
        print(f"\n⚠ 警告: {split_name} 的 labels 目录为空！")
        return False
    
    # 检查匹配情况
    image_stems = {f.stem for f in image_files}
    label_stems = {f.stem for f in label_files}
    
    matched = image_stems & label_stems
    only_images = image_stems - label_stems
    only_labels = label_stems - image_stems
    
    print(f"\n匹配情况:")
    print(f"  - 图片和标签匹配: {len(matched)} 对")
    if only_images:
        print(f"  - 只有图片没有标签: {len(only_images)} 个 (前5个: {list(only_images)[:5]})")
    if only_labels:
        print(f"  - 只有标签没有图片: {len(only_labels)} 个 (前5个: {list(only_labels)[:5]})")
    
    if len(matched) == 0:
        print(f"\n✗ 错误: 没有找到匹配的图片-标签对！")
        return False
    
    print(f"✓ {split_name} 数据集检查通过！")
    return True

def main():
    base_path = Path(__file__).parent / "VisDrone2YOLO"
    
    if not base_path.exists():
        print(f"✗ VisDrone2YOLO 目录不存在: {base_path}")
        return
    
    print("VisDrone 数据集完整性检查")
    print(f"基础路径: {base_path}")
    
    results = []
    for split in ['train', 'val', 'test-dev']:
        results.append(check_dataset_split(split, base_path))
    
    print(f"\n{'='*60}")
    print("检查总结")
    print(f"{'='*60}")
    if all(results):
        print("✓ 所有数据集分割检查通过，可以开始训练！")
    else:
        print("✗ 部分数据集分割存在问题，请先解决后再训练")
        print("\n下载 VisDrone 图片的说明:")
        print("1. 访问: https://github.com/VisDrone/VisDrone-Dataset")
        print("2. 下载对应的图片压缩包:")
        print("   - VisDrone2019-DET-train.zip (训练集图片)")
        print("   - VisDrone2019-DET-val.zip (验证集图片)")
        print("   - VisDrone2019-DET-test-dev.zip (测试集图片)")
        print("3. 解压后，将图片文件（.jpg）放入对应的 images 目录")

if __name__ == "__main__":
    main()

