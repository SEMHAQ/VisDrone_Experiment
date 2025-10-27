#!/usr/bin/env python3
"""
VisDrone标注格式转YOLO格式转换器
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.path_manager import path_manager


class AnnotationConverter:
    """标注格式转换器"""

    def __init__(self):
        # VisDrone类别到YOLO类别的映射（忽略0和11类别）
        self.category_mapping = {
            1: 0,  # pedestrian -> 0
            2: 1,  # people -> 1
            3: 2,  # bicycle -> 2
            4: 3,  # car -> 3
            5: 4,  # van -> 4
            6: 5,  # truck -> 5
            7: 6,  # tricycle -> 6
            8: 7,  # awning-tricycle -> 7
            9: 8,  # bus -> 8
            10: 9,  # motor -> 9
            # 忽略 0: ignore, 11: others
        }

        # YOLO类别名称（按映射后的顺序）
        self.yolo_class_names = [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]

    def convert_annotation(self, annotation_path, image_width, image_height):
        """转换单个标注文件"""
        yolo_annotations = []

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            print(f"❌ 无法读取标注文件: {annotation_path}")
            return yolo_annotations

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 8:
                continue

            try:
                # 解析VisDrone格式
                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_width = float(parts[2])
                bbox_height = float(parts[3])
                object_category = int(parts[5])

                # 过滤无效目标
                if object_category not in self.category_mapping:
                    continue

                # 过滤小目标（可选）
                if bbox_width < 2 or bbox_height < 2:
                    continue

                # 转换为YOLO格式
                x_center = (bbox_left + bbox_width / 2) / image_width
                y_center = (bbox_top + bbox_height / 2) / image_height
                width = bbox_width / image_width
                height = bbox_height / image_height

                # 确保坐标在0-1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                # 跳过无效的边界框
                if width <= 0 or height <= 0:
                    continue

                # 获取YOLO类别ID
                yolo_class_id = self.category_mapping[object_category]

                # 添加到结果
                yolo_annotations.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            except (ValueError, IndexError) as e:
                print(f"⚠ 解析错误 {annotation_path}: {line} -> {e}")
                continue

        return yolo_annotations

    def get_image_size(self, image_path):
        """获取图像尺寸"""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            print(f"❌ 无法获取图像尺寸 {image_path}: {e}")
            return None

    def convert_dataset_split(self, split_name):
        """转换整个数据集划分（train/val/test）"""
        print(f"\n🔄 转换 {split_name} 数据集...")

        # 原始标注目录
        orig_annotations_dir = path_manager.dataset_root / "annotations" / split_name
        # 原始图像目录
        orig_images_dir = path_manager.dataset_root / "images" / split_name
        # 转换后的标注目录
        yolo_annotations_dir = path_manager.dataset_root / "labels" / split_name

        # 创建输出目录
        yolo_annotations_dir.mkdir(parents=True, exist_ok=True)

        if not orig_annotations_dir.exists():
            print(f"❌ 原始标注目录不存在: {orig_annotations_dir}")
            return False

        if not orig_images_dir.exists():
            print(f"❌ 原始图像目录不存在: {orig_images_dir}")
            return False

        # 获取所有标注文件
        annotation_files = list(orig_annotations_dir.glob("*.txt"))
        if not annotation_files:
            print(f"❌ 在 {split_name} 中没有找到标注文件")
            return False

        print(f"📁 找到 {len(annotation_files)} 个标注文件")

        success_count = 0
        error_count = 0

        # 转换每个标注文件
        for annotation_file in tqdm(annotation_files, desc=f"转换 {split_name}"):
            try:
                # 对应的图像文件
                image_name = annotation_file.stem + ".jpg"
                image_path = orig_images_dir / image_name

                if not image_path.exists():
                    # 尝试其他图像格式
                    for ext in ['.jpg', '.png', '.jpeg']:
                        alt_image_path = orig_images_dir / (annotation_file.stem + ext)
                        if alt_image_path.exists():
                            image_path = alt_image_path
                            break

                if not image_path.exists():
                    print(f"⚠ 找不到图像文件: {image_name}")
                    error_count += 1
                    continue

                # 获取图像尺寸
                image_size = self.get_image_size(image_path)
                if image_size is None:
                    error_count += 1
                    continue

                image_width, image_height = image_size

                # 转换标注
                yolo_annotations = self.convert_annotation(annotation_file, image_width, image_height)

                # 保存转换后的标注
                output_file = yolo_annotations_dir / annotation_file.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_annotations))

                success_count += 1

            except Exception as e:
                print(f"❌ 转换失败 {annotation_file}: {e}")
                error_count += 1
                continue

        print(f"✅ {split_name} 转换完成: {success_count} 成功, {error_count} 失败")
        return success_count > 0

    def create_dataset_yaml(self):


        config_content = f"""# YOLO格式的VisDrone数据集配置
path: {path_manager.dataset_root}  # 数据集根目录

# 图像和标注路径
train: images/train
val: images/val
test: images/test_dev

# 标注目录（转换后的YOLO格式）
train_labels: labels/train
val_labels: labels/val
test_labels: labels/test_dev

# 类别数量
nc: {len(self.yolo_class_names)}

# 类别名称
names: {self.yolo_class_names}

# 训练参数
img_size: 640
batch_size: 16
epochs: 80
"""

        with open(yolo_dataset_config, 'w', encoding='utf-8') as f:
            f.write(config_content)

        print(f"✅ 创建YOLO数据集配置: {yolo_dataset_config}")
        return yolo_dataset_config

    def convert_full_dataset(self):
        """转换整个数据集"""
        print("🚀 开始转换VisDrone数据集为YOLO格式")
        print("=" * 50)

        # 验证原始数据集结构
        if not self.validate_original_dataset():
            return False

        # 转换各个划分
        splits = ['train', 'val']
        # 注意：test_dev的标注通常不公开，所以可能无法转换

        all_success = True
        for split in splits:
            if not self.convert_dataset_split(split):
                all_success = False

        if all_success:
            # 创建YOLO格式的配置文件
            # self.create_dataset_yaml()
            print("\n🎉 数据集转换完成!")
            print(f"📁 转换后的标注保存在: {path_manager.dataset_root / 'labels'}")

        else:
            print("\n⚠ 数据集转换过程中出现错误，请检查上述输出")

        return all_success

    def validate_original_dataset(self):
        """验证原始数据集结构"""
        print("🔍 验证原始数据集结构...")

        required_dirs = [
            path_manager.dataset_root / "annotations" / "train",
            path_manager.dataset_root / "annotations" / "val",
            path_manager.dataset_root / "images" / "train",
            path_manager.dataset_root / "images" / "val"
        ]

        missing_dirs = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))

        if missing_dirs:
            print("❌ 缺少必要的目录:")
            for dir_path in missing_dirs:
                print(f"   - {dir_path}")
            return False

        # 检查文件数量
        train_ann_count = len(list((path_manager.dataset_root / "annotations" / "train").glob("*.txt")))
        val_ann_count = len(list((path_manager.dataset_root / "annotations" / "val").glob("*.txt")))

        print(f"✅ 训练集标注文件: {train_ann_count} 个")
        print(f"✅ 验证集标注文件: {val_ann_count} 个")

        if train_ann_count == 0 or val_ann_count == 0:
            print("❌ 标注文件数量为0，请检查数据集")
            return False

        return True

    def statistics(self):
        """统计数据集信息"""
        print("\n📊 数据集统计信息")
        print("=" * 30)

        for split in ['train', 'val']:
            labels_dir = path_manager.dataset_root / "labels" / split
            if not labels_dir.exists():
                print(f"❌ {split} 的标签目录不存在")
                continue

            label_files = list(labels_dir.glob("*.txt"))
            total_objects = 0
            class_counts = {i: 0 for i in range(len(self.yolo_class_names))}

            for label_file in label_files:
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_objects += len(lines)
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if class_id in class_counts:
                                class_counts[class_id] += 1

            print(f"\n{split.upper()} 集:")
            print(f"  图像数量: {len(label_files)}")
            print(f"  目标总数: {total_objects}")
            print(f"  各类别数量:")
            for class_id, count in class_counts.items():
                if count > 0:
                    print(f"    {self.yolo_class_names[class_id]}: {count}")


def convert_visdrone_to_yolo():
    """转换VisDrone数据集为YOLO格式"""
    converter = AnnotationConverter()
    return converter.convert_full_dataset()


def show_dataset_statistics():
    """显示数据集统计信息"""
    converter = AnnotationConverter()
    converter.statistics()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='VisDrone标注格式转换器')
    parser.add_argument('--convert', action='store_true', help='转换数据集')
    parser.add_argument('--stats', action='store_true', help='显示统计信息')

    args = parser.parse_args()

    if args.convert:
        convert_visdrone_to_yolo()
    elif args.stats:
        show_dataset_statistics()
    else:
        print("用法:")
        print("  python annotation_converter.py --convert   # 转换数据集")
        print("  python annotation_converter.py --stats     # 显示统计信息")