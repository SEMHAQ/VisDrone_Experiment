#!/usr/bin/env python3
"""
项目初始化脚本
"""

import os
import sys
from pathlib import Path


def init_project():
    """初始化项目结构"""
    project_root = Path(__file__).parent.parent

    # 创建必要目录
    dirs_to_create = [
        'configs/experiments',
        'models/modules',
        'scripts',
        'utils',
        'results'
    ]

    for dir_path in dirs_to_create:
        (project_root / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")

    # 创建必要的空文件
    files_to_create = [
        'models/__init__.py',
        'models/modules/__init__.py',
        'scripts/__init__.py',
        'utils/__init__.py'
    ]

    for file_path in files_to_create:
        (project_root / file_path).touch()
        print(f"创建文件: {file_path}")

    print("\n项目结构初始化完成!")


if __name__ == "__main__":
    init_project()