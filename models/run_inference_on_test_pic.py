#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 models 目录下的两个 best.pt 模型，对 models/test_pic 中的图片批量推理，
并将带框结果保存在 models 目录下，按模型名分文件夹：
- models/pred_best_baseline/
- models/pred_best_final/

运行：
    python models/run_inference_on_test_pic.py
"""

from pathlib import Path
import sys


def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        return 'cpu'
    except Exception:
        return 'cpu'


def run_for_model(model_path: Path, source_dir: Path, out_project: Path):
    from ultralytics import YOLO

    if not model_path.exists():
        print(f"错误：模型文件不存在 -> {model_path}")
        return

    if not source_dir.exists():
        print(f"错误：图片目录不存在 -> {source_dir}")
        return

    images = []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for p in source_dir.iterdir():
        if p.suffix.lower() in exts and p.is_file():
            images.append(p)
    if not images:
        print(f"错误：图片目录为空或无支持格式 -> {source_dir}")
        return

    print(f"\n加载模型: {model_path}")
    model = YOLO(str(model_path))

    device = detect_device()
    out_name = f"pred_{model_path.stem}"

    predict_args = {
        'source': str(source_dir),
        'imgsz': 1024,
        'conf': 0.25,
        'iou': 0.5,
        'device': device,
        'save': True,
        'save_txt': False,
        'save_conf': True,
        'project': str(out_project),
        'name': out_name,
        'exist_ok': True,
    }

    print("开始推理...")
    for k, v in predict_args.items():
        print(f"  {k}: {v}")

    results = model.predict(**predict_args)

    save_dir = None
    if results and hasattr(results[0], 'save_dir'):
        save_dir = Path(results[0].save_dir)
    else:
        # Ultralytics 会创建输出目录，即使返回值里没有 save_dir
        save_dir = out_project / out_name

    print(f"✓ 推理完成！输出目录: {save_dir}")

    try:
        total_boxes = 0
        for r in results:
            n = len(getattr(r, 'boxes', []))
            total_boxes += n
        print(f"统计：处理图片数={len(images)}，总检测框数={total_boxes}")
    except Exception:
        pass


def main():
    base_dir = Path(__file__).resolve().parent
    models = [
        base_dir / "best_baseline.pt",
        base_dir / "best_final.pt",
    ]
    source_dir = base_dir / "test_pic"
    out_project = base_dir

    print(f"模型目录: {base_dir}")
    print(f"图片目录: {source_dir}")
    for m in models:
        run_for_model(m, source_dir, out_project)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("运行失败：", e)
        sys.exit(1)