#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估脚本 - 评估训练好的 YOLOv8s 模型
"""
import os
import sys
from pathlib import Path

def main():
    # 确保在项目根目录
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    print(f"工作目录: {project_root}")

    from ultralytics import YOLO

    # 查找最新的训练权重
    weights_dir = project_root / "runs" / "visdrone"
    
    # 如果指定了模型路径，使用指定的；否则自动找最新的
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
        if not model_path.exists():
            print(f"错误: 指定的模型路径不存在: {model_path}")
            sys.exit(1)
    else:
        # 自动查找 baseline 实验的最佳权重
        baseline_dir = weights_dir / "baseline_y8s_1024_adamw"
        if baseline_dir.exists():
            best_weights = baseline_dir / "weights" / "best.pt"
            if best_weights.exists():
                model_path = best_weights
                print(f"找到训练权重: {model_path}")
            else:
                print(f"错误: 找不到最佳权重文件: {best_weights}")
                print("请指定模型路径，例如: python eval_model.py runs/visdrone/baseline_y8s_1024_adamw/weights/best.pt")
                sys.exit(1)
        else:
            print(f"错误: 找不到训练目录: {baseline_dir}")
            print("请指定模型路径，例如: python eval_model.py runs/visdrone/baseline_y8s_1024_adamw/weights/best.pt")
            sys.exit(1)

    # 加载模型
    print(f"\n正在加载模型: {model_path}")
    try:
        model = YOLO(str(model_path))
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 数据配置文件
    data_yaml = project_root / "cfg" / "visdrone.yaml"
    if not data_yaml.exists():
        print(f"错误: 数据配置文件不存在: {data_yaml}")
        sys.exit(1)

    # 评估参数
    eval_args = {
        'data': str(data_yaml),
        'imgsz': 1024,
        'batch': 1,  # 评估时 batch=1 确保结果稳定
        'conf': 0.25,  # 置信度阈值
        'iou': 0.7,   # NMS IoU 阈值
        'device': 0,
        'verbose': True,
        'save_json': False,  # 设置为 True 可以保存 JSON 格式的评估结果
        'save_hybrid': False,  # 保存混合标签（预测+真实标签）
    }

    print("\n开始评估...")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"数据: {data_yaml}")
    print(f"图像尺寸: {eval_args['imgsz']}")
    print(f"置信度阈值: {eval_args['conf']}")
    print(f"IoU 阈值: {eval_args['iou']}")
    print("=" * 60)

    try:
        # 运行验证
        results = model.val(**eval_args)
        
        print("\n" + "=" * 60)
        print("评估结果摘要")
        print("=" * 60)
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"\n每个类别的 mAP50-95:")
        if hasattr(results, 'names'):
            for i, name in enumerate(results.names.values()):
                if i < len(results.box.maps):
                    print(f"  {name}: {results.box.maps[i]:.4f}")
        
        print(f"\n详细结果保存在: {results.save_dir}")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

