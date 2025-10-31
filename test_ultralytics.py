#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 ultralytics 是否正常工作"""

import sys
print("Python 版本:", sys.version)
print("Python 路径:", sys.executable)

try:
    import ultralytics
    print("✓ ultralytics 已安装, 版本:", ultralytics.__version__)
except ImportError as e:
    print("✗ ultralytics 未安装或导入失败:", e)
    sys.exit(1)

try:
    import torch
    print("✓ torch 已安装, 版本:", torch.__version__)
    print("✓ CUDA 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("✓ GPU 设备:", torch.cuda.get_device_name(0))
except ImportError as e:
    print("✗ torch 未安装或导入失败:", e)
    sys.exit(1)

print("\n测试 ultralytics YOLO 基础功能...")
try:
    from ultralytics import YOLO
    model = YOLO('yolov8s.pt')
    print("✓ YOLOv8s 模型加载成功")
except Exception as e:
    print("✗ 模型加载失败:", e)
    import traceback
    traceback.print_exc()

print("\n测试完成！")

