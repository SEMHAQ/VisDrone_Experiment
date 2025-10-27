#!/usr/bin/env python3
"""
消融实验框架初始化脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from experiments.ablation_manager import ablation_manager


def main():
    """初始化消融实验框架"""
    print("🛠 初始化消融实验框架")
    print("=" * 50)

    # 设置所有实验
    ablation_manager.setup_all_experiments()

    print("\n🎉 消融实验框架初始化完成!")
    print("\n下一步操作:")
    print("1. 检查 configs/experiments/ 目录下的配置文件")
    print("2. 运行 python main.py 开始实验")
    print("3. 选择 '训练基准模型' 开始第一个实验")


if __name__ == "__main__":
    main()