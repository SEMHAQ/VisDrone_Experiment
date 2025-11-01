#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比评估脚本 - 对比基线模型和 P2 模型的性能
"""
import os
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    from ultralytics import YOLO
    
    # 数据配置文件
    data_yaml = project_root / "cfg" / "visdrone.yaml"
    
    # 查找模型权重
    baseline_weights = project_root / "runs" / "visdrone" / "baseline_y8s_1024_adamw" / "weights" / "best.pt"
    p2_weights = project_root / "runs" / "visdrone" / "y8s_p2_1024_adamw_bs4" / "weights" / "best.pt"
    
    # 检查权重文件是否存在
    models_to_eval = []
    
    if baseline_weights.exists():
        models_to_eval.append(("Baseline (YOLOv8s)", baseline_weights))
    else:
        print(f"⚠ 警告: 基线模型权重不存在: {baseline_weights}")
    
    if p2_weights.exists():
        models_to_eval.append(("P2 Model (YOLOv8s-P2)", p2_weights))
    else:
        print(f"⚠ 警告: P2 模型权重不存在: {p2_weights}")
        # 尝试查找其他可能的路径
        p2_dirs = list((project_root / "runs" / "visdrone").glob("y8s_p2*"))
        if p2_dirs:
            p2_weights = p2_dirs[0] / "weights" / "best.pt"
            if p2_weights.exists():
                models_to_eval.append(("P2 Model", p2_weights))
                print(f"找到 P2 模型: {p2_weights}")
    
    if len(models_to_eval) == 0:
        print("错误: 找不到任何模型权重文件")
        sys.exit(1)
    
    results_summary = []
    
    print("=" * 70)
    print("模型性能对比评估")
    print("=" * 70)
    
    for model_name, model_path in models_to_eval:
        print(f"\n评估模型: {model_name}")
        print(f"权重路径: {model_path}")
        print("-" * 70)
        
        try:
            model = YOLO(str(model_path))
            
            # 运行评估
            eval_results = model.val(
                data=str(data_yaml),
                imgsz=1024,
                batch=1,
                conf=0.25,
                iou=0.7,
                device=0,
                verbose=True,
                save_json=False,
            )
            
            # 提取关键指标
            summary = {
                'model_name': model_name,
                'map50': eval_results.box.map50,
                'map50_95': eval_results.box.map,
                'precision': eval_results.box.p[0] if hasattr(eval_results.box, 'p') else 0,
                'recall': eval_results.box.r[0] if hasattr(eval_results.box, 'r') else 0,
                'f1': eval_results.box.f[0] if hasattr(eval_results.box, 'f') else 0,
            }
            
            # 提取每个类别的 AP
            if hasattr(eval_results.box, 'maps') and len(eval_results.box.maps) > 0:
                class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 
                             'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
                summary['class_aps'] = {}
                for i, name in enumerate(class_names):
                    if i < len(eval_results.box.maps):
                        summary['class_aps'][name] = eval_results.box.maps[i]
            
            results_summary.append(summary)
            
            print(f"✓ mAP50: {summary['map50']:.4f}")
            print(f"✓ mAP50-95: {summary['map50_95']:.4f}")
            print(f"✓ Precision: {summary['precision']:.4f}")
            print(f"✓ Recall: {summary['recall']:.4f}")
            print(f"✓ F1: {summary['f1']:.4f}")
            
        except Exception as e:
            print(f"✗ 评估失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 对比结果
    if len(results_summary) >= 2:
        print("\n" + "=" * 70)
        print("性能对比总结")
        print("=" * 70)
        
        baseline = results_summary[0]
        p2 = results_summary[1]
        
        print(f"\n{'指标':<20} {'Baseline':<15} {'P2 Model':<15} {'提升':<15}")
        print("-" * 70)
        
        metrics = [
            ('mAP50', 'map50'),
            ('mAP50-95', 'map50_95'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
            ('F1-Score', 'f1'),
        ]
        
        for metric_name, metric_key in metrics:
            baseline_val = baseline[metric_key]
            p2_val = p2[metric_key]
            improvement = p2_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            
            print(f"{metric_name:<20} {baseline_val:<15.4f} {p2_val:<15.4f} {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        # 小目标类别对比
        if 'class_aps' in baseline and 'class_aps' in p2:
            print("\n" + "-" * 70)
            print("小目标类别详细对比 (mAP50-95)")
            print("-" * 70)
            
            small_target_classes = ['pedestrian', 'people', 'bicycle', 'motor', 'tricycle', 'awning-tricycle']
            
            for cls in small_target_classes:
                if cls in baseline['class_aps'] and cls in p2['class_aps']:
                    baseline_ap = baseline['class_aps'][cls]
                    p2_ap = p2['class_aps'][cls]
                    improvement = p2_ap - baseline_ap
                    improvement_pct = (improvement / baseline_ap * 100) if baseline_ap > 0 else 0
                    
                    print(f"{cls:<20} {baseline_ap:<15.4f} {p2_ap:<15.4f} {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        print("\n" + "=" * 70)
        print("结论:")
        overall_improvement = p2['map50_95'] - baseline['map50_95']
        if overall_improvement > 0.01:
            print(f"✓ P2 模型相比基线有显著提升 (+{overall_improvement:.4f} mAP50-95)")
        elif overall_improvement > 0:
            print(f"✓ P2 模型相比基线有轻微提升 (+{overall_improvement:.4f} mAP50-95)")
        else:
            print(f"⚠ P2 模型相比基线下降了 ({overall_improvement:.4f} mAP50-95)")
        print("=" * 70)
    
    # 保存结果到文件
    output_file = project_root / "results" / "p2_comparison_summary.md"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# P2 模型性能对比\n\n")
        f.write("## 评估结果\n\n")
        
        for result in results_summary:
            f.write(f"### {result['model_name']}\n\n")
            f.write(f"- mAP50: {result['map50']:.4f}\n")
            f.write(f"- mAP50-95: {result['map50_95']:.4f}\n")
            f.write(f"- Precision: {result['precision']:.4f}\n")
            f.write(f"- Recall: {result['recall']:.4f}\n")
            f.write(f"- F1: {result['f1']:.4f}\n\n")
            
            if 'class_aps' in result:
                f.write("#### 各类别 mAP50-95\n\n")
                for cls, ap in result['class_aps'].items():
                    f.write(f"- {cls}: {ap:.4f}\n")
                f.write("\n")
        
        if len(results_summary) >= 2:
            f.write("## 对比分析\n\n")
            f.write(f"总体提升: {p2['map50_95'] - baseline['map50_95']:+.4f} mAP50-95\n")
    
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == '__main__':
    main()

