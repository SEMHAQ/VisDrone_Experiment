#!/usr/bin/env python3
"""
环境验证脚本 - 验证所有路径和依赖是否正确设置
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.path_manager import path_manager
import subprocess


def check_dependencies():
    """检查必要的依赖包"""
    print("🔍 检查依赖包...")

    required_packages = [
        'ultralytics',
        'torch',
        'opencv-python',
        'matplotlib',
        'seaborn',
        'pandas',
        'tqdm'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")

    if missing_packages:
        print(f"\n⚠ 缺少以下包: {missing_packages}")
        install = input("是否自动安装？ (y/n): ")
        if install.lower() == 'y':
            for package in missing_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            print("请手动安装缺少的包")
            return False

    return True


def check_dataset_structure():
    """检查数据集结构"""
    print("\n📁 检查数据集结构...")

    required_dirs = [
        path_manager.dataset_root / "images" / "train",
        path_manager.dataset_root / "images" / "val",
        path_manager.dataset_root / "images" / "test_dev",
        path_manager.dataset_root / "annotations" / "train",
        path_manager.dataset_root / "annotations" / "val"
    ]

    all_exists = True
    for dir_path in required_dirs:
        if dir_path.exists():
            # 统计文件数量
            file_count = len(list(dir_path.glob("*")))
            print(f"   ✅ {dir_path.relative_to(path_manager.project_root)} ({file_count} 个文件)")
        else:
            print(f"   ❌ {dir_path.relative_to(path_manager.project_root)}")
            all_exists = False

    return all_exists


def check_config_files():
    """检查配置文件"""
    print("\n⚙ 检查配置文件...")

    if path_manager.dataset_config.exists():
        print(f"   ✅ {path_manager.dataset_config.relative_to(path_manager.project_root)}")
        return True
    else:
        print(f"   ❌ {path_manager.dataset_config.relative_to(path_manager.project_root)}")
        return False


def main():
    print("=" * 60)
    print("        VisDrone实验环境验证")
    print("=" * 60)

    # 检查依赖
    if not check_dependencies():
        return False

    # 验证路径
    print(f"📂 项目根目录: {path_manager.project_root}")

    if not path_manager.validate_paths():
        print("\n❌ 环境验证失败，请检查上述错误")
        return False



    # 检查数据集
    if not check_dataset_structure():
        return False

    # 检查配置
    if not check_config_files():
        return False

    print("\n🎉 环境验证完成！所有设置正确。")
    print("\n下一步:")
    print("1. 运行 'python scripts/train_baseline.py' 开始训练基准模型")
    print("2. 查看 'configs/dataset/visdrone.yaml' 确认配置")

    return True


if __name__ == "__main__":
    main()