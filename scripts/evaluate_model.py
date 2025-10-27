#!/usr/bin/env python3
"""
模型评估脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from ultralytics import YOLO
from utils.path_manager import path_manager


def find_latest_model(exp_name="baseline"):
    """查找指定实验的最新模型"""
    exp_dir = path_manager.get_experiment_dir(exp_name)
    weights_dir = exp_dir / "weights"

    if not weights_dir.exists():
        print(f"❌ 实验目录不存在: {weights_dir}")
        return None

    # 查找最佳模型
    best_model = weights_dir / "best.pt"
    if best_model.exists():
        return best_model

    # 查找最后一个模型
    last_model = weights_dir / "last.pt"
    if last_model.exists():
        return last_model

    print(f"❌ 在 {exp_name} 中找不到模型文件")
    return None


def evaluate_model(exp_name="baseline"):
    """评估指定实验的模型"""

    print(f"📊 开始评估模型: {exp_name}")
    print("=" * 50)

    # 查找模型
    model_path = find_latest_model(exp_name)
    if model_path is None:
        return

    print(f"🔍 找到模型: {model_path}")

    # 验证配置文件
    if not path_manager.dataset_config.exists():
        print(f"❌ 配置文件不存在: {path_manager.dataset_config}")
        return

    # 加载模型
    print("🔄 加载模型...")
    model = YOLO(str(model_path))

    # 在验证集上评估
    print("🧪 在验证集上评估模型...")
    metrics = model.val(
        data=str(path_manager.dataset_config),
        split='val',
        imgsz=640,
        batch=16,
        conf=0.001,  # 低置信度阈值以评估召回率
        iou=0.6,
        device=0,
        verbose=True
    )

    # 打印结果
    print(f"\n🎯 {exp_name} 模型性能:")
    print(f"   mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   精确率:      {metrics.box.precision.mean():.4f}")
    print(f"   召回率:      {metrics.box.recall.mean():.4f}")

    # 保存结果到文件
    results_dir = path_manager.get_experiment_dir(exp_name) / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "evaluation_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"{exp_name} 模型评估结果\n")
        f.write("=" * 40 + "\n")
        f.write(f"评估时间: {metrics.speed['inference']:.2f} ms/img\n")
        f.write(f"mAP@0.5:     {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
        f.write(f"精确率:      {metrics.box.precision.mean():.4f}\n")
        f.write(f"召回率:      {metrics.box.recall.mean():.4f}\n")
        f.write(f"参数数量:    {metrics.model.nparams if hasattr(metrics, 'model') else 'N/A'}\n")

    print(f"✅ 评估结果已保存到: {results_file}")

    return metrics


def compare_models(model_paths, model_names):
    """比较多个模型的性能"""
    print("📈 模型性能对比")
    print("=" * 50)

    results = {}
    for path, name in zip(model_paths, model_names):
        if path.exists():
            model = YOLO(str(path))
            metrics = model.val(
                data=str(path_manager.dataset_config),
                split='val',
                imgsz=640,
                batch=16,
                verbose=False
            )
            results[name] = {
                'mAP50': metrics.box.map50,
                'mAP': metrics.box.map,
                'precision': metrics.box.precision.mean(),
                'recall': metrics.box.recall.mean()
            }
            print(f"✅ {name}: mAP50 = {metrics.box.map50:.4f}")
        else:
            print(f"❌ {name}: 模型文件不存在")

    return results


if __name__ == "__main__":
    # 默认评估基准模型
    evaluate_model("baseline")