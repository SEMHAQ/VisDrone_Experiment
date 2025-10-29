#!/usr/bin/env python3
"""
VisDrone实验主入口
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def main():
    print("=" * 50)
    print("       VisDrone目标检测实验")
    print("=" * 50)

    while True:
        print("\n请选择操作:")
        print("1. 验证环境")
        print("2. 训练基准模型")
        print("3. 运行消融实验")
        print("4. 评估模型")
        print("5. 退出")

        choice = input("\n请输入选择 (1-5): ").strip()

        if choice == "1":
            verify_environment()
        elif choice == "2":
            train_baseline()
        elif choice == "3":
            run_ablation_study()
        elif choice == "4":
            evaluate_models()
        elif choice == "5":
            print("再见!")
            break
        else:
            print("无效选择!")

        input("\n按Enter键继续...")


def verify_environment():
    """验证环境"""
    print("\n验证环境中...")
    os.system("python scripts/verify_environment.py")


def train_baseline():
    """训练基准模型"""
    print("\n训练基准模型中...")
    os.system("python scripts/train_baseline.py")


def run_ablation_study():
    """运行消融实验"""
    print("\n运行消融实验中...")
    os.system("python scripts/run_ablation_study.py")


def evaluate_models():
    """评估模型"""
    print("\n评估模型中...")
    os.system("python scripts/evaluate_model.py")


if __name__ == "__main__":
    main()