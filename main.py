#!/usr/bin/env python3
"""
VisDrone实验主入口 - 更新版本
"""

import os
import sys
import subprocess


def main():
    print("=" * 50)
    print("       VisDrone目标检测实验")
    print("=" * 50)

    while True:
        print("\n请选择操作:")
        print("1. 🔍 验证环境")
        print("2. 🎯 训练基准模型")
        print("3. 📊 评估基准模型")
        print("4. 🧪 运行消融实验")
        print("5. 📈 结果分析")
        print("6. ❌ 退出")

        choice = input("\n请输入选择 (1-6): ").strip()

        if choice == "1":
            print("\n验证环境...")
            os.system("python scripts/verify_environment.py")

        elif choice == "2":
            print("\n开始训练基准模型...")
            os.system("python scripts/train_baseline.py")

        elif choice == "3":
            print("\n评估基准模型...")
            os.system("python scripts/evaluate_model.py")

        elif choice == "4":
            print("\n运行消融实验...")
            run_ablation_study()

        elif choice == "5":
            print("\n结果分析...")
            run_analysis()

        elif choice == "6":
            print("👋 再见!")
            break

        else:
            print("❌ 无效选择!")

        input("\n按Enter键继续...")


def run_ablation_study():
    """运行消融实验"""
    print("消融实验功能开发中...")
    print("当前可手动运行以下实验:")
    print("1. 基准模型: python scripts/train_baseline.py")
    print("2. 评估模型: python scripts/evaluate_model.py")


def run_analysis():
    """运行结果分析"""
    print("结果分析功能开发中...")
    print("请检查以下目录中的结果:")
    print("- runs/baseline/metrics/")
    print("- 训练日志和评估结果")


if __name__ == "__main__":
    main()