#!/usr/bin/env python3
"""
标注转换脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.annotation_converter import convert_visdrone_to_yolo, show_dataset_statistics


def main():
    print("🔄 VisDrone标注格式转换工具")
    print("=" * 50)

    print("请选择操作:")
    print("1. 🔄 转换数据集为YOLO格式")
    print("2. 📊 显示数据集统计信息")
    print("3. ❌ 退出")

    choice = input("\n请输入选择 (1-3): ").strip()

    if choice == "1":
        print("\n开始转换数据集...")
        success = convert_visdrone_to_yolo()

        if success:
            print("\n🎉 转换完成!")
            print("\n下一步:")
            print("1. 更新 configs/dataset/visdrone.yaml 使用新的标注路径")
            print("2. 运行训练脚本开始实验")
        else:
            print("\n❌ 转换失败，请检查错误信息")

    elif choice == "2":
        print("\n生成数据集统计信息...")
        show_dataset_statistics()

    elif choice == "3":
        print("👋 再见!")
        return

    else:
        print("❌ 无效选择!")


if __name__ == "__main__":
    main()