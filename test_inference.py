#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理测试脚本 - 在单张图片或测试集上运行推理
"""
import os
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    from ultralytics import YOLO
    
    # 查找模型权重
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
    else:
        baseline_dir = project_root / "runs" / "visdrone" / "baseline_y8s_1024_adamw"
        model_path = baseline_dir / "weights" / "best.pt"
        if not model_path.exists():
            print(f"错误: 找不到模型: {model_path}")
            print("用法: python test_inference.py <模型路径> [图片路径或目录]")
            sys.exit(1)
    
    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path))
    
    # 确定输入源
    if len(sys.argv) > 2:
        source = sys.argv[2]
    else:
        # 默认使用验证集的一张图片
        val_images = project_root / "VisDrone2YOLO" / "VisDrone2019-DET-val" / "images"
        images = list(val_images.glob("*.jpg"))
        if images:
            source = str(images[0])
            print(f"使用验证集图片: {source}")
        else:
            print("错误: 找不到测试图片")
            sys.exit(1)
    
    # 推理参数
    predict_args = {
        'source': source,
        'imgsz': 1024,
        'conf': 0.25,
        'iou': 0.5,
        'device': 0,
        'save': True,
        'save_txt': False,  # 是否保存检测框坐标
        'save_conf': True,  # 是否保存置信度
        'show_labels': True,
        'show_conf': True,
    }
    
    print("\n开始推理...")
    print("=" * 60)
    for key, value in predict_args.items():
        print(f"{key}: {value}")
    print("=" * 60)
    
    results = model.predict(**predict_args)
    
    print(f"\n✓ 推理完成！")
    print(f"结果保存在: {results[0].save_dir if results else '未知'}")
    
    # 显示检测到的目标数量
    for i, result in enumerate(results):
        print(f"\n图片 {i+1}: 检测到 {len(result.boxes)} 个目标")
        if len(result.boxes) > 0:
            print("前5个检测结果:")
            for j, box in enumerate(result.boxes[:5]):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = result.names[cls_id]
                print(f"  {j+1}. {cls_name}: {conf:.3f}")

if __name__ == '__main__':
    main()

