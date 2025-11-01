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
    
    # 查找模型权重（支持多个模型对比）
    models_to_eval = []
    
    # Baseline 模型
    baseline_weights = project_root / "runs" / "visdrone" / "baseline_y8s_1024_adamw" / "weights" / "best.pt"
    if baseline_weights.exists():
        models_to_eval.append(("Baseline (YOLOv8s)", baseline_weights))
    else:
        print(f"⚠ 警告: 基线模型权重不存在: {baseline_weights}")
    
    # P2 模型
    p2_weights = project_root / "runs" / "visdrone" / "y8s_p2_1024_adamw_bs4" / "weights" / "best.pt"
    if not p2_weights.exists():
        # 尝试查找其他可能的路径
        p2_dirs = list((project_root / "runs" / "visdrone").glob("y8s_p2*"))
        p2_dirs = [d for d in p2_dirs if 'bifpn' not in d.name.lower()]  # 排除 BiFPN
        if p2_dirs:
            p2_weights = p2_dirs[0] / "weights" / "best.pt"
    
    if p2_weights.exists():
        models_to_eval.append(("P2 Model (YOLOv8s-P2)", p2_weights))
    else:
        print(f"⚠ 警告: P2 模型权重不存在")
    
    # P2 + BiFPN 模型
    bifpn_weights = project_root / "runs" / "visdrone" / "y8s_p2_bifpn_1024_adamw" / "weights" / "best.pt"
    if not bifpn_weights.exists():
        # 尝试查找其他可能的路径
        bifpn_dirs = list((project_root / "runs" / "visdrone").glob("*bifpn*"))
        if bifpn_dirs:
            bifpn_weights = bifpn_dirs[0] / "weights" / "best.pt"
    
    if bifpn_weights.exists():
        models_to_eval.append(("P2+BiFPN (YOLOv8s-P2-BiFPN)", bifpn_weights))
    else:
        print(f"⚠ 提示: P2+BiFPN 模型权重不存在（可能还在训练中）")
    
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
        other_models = results_summary[1:]
        
        # 构建表头
        header = f"{'指标':<20}"
        for model_name, _ in models_to_eval:
            short_name = model_name.split('(')[0].strip() if '(' in model_name else model_name[:12]
            header += f" {short_name:<18}"
        if len(other_models) > 0:
            header += f" {'vs Baseline':<15}"
        print(header)
        print("-" * 70)
        
        metrics = [
            ('mAP50', 'map50'),
            ('mAP50-95', 'map50_95'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
            ('F1-Score', 'f1'),
        ]
        
        for metric_name, metric_key in metrics:
            row = f"{metric_name:<20}"
            baseline_val = baseline[metric_key]
            row += f" {baseline_val:<18.4f}"
            
            for other_model in other_models:
                other_val = other_model[metric_key]
                row += f" {other_val:<18.4f}"
            
            # 计算相对基线的提升
            if len(other_models) > 0:
                best_val = max([m[metric_key] for m in other_models])
                improvement = best_val - baseline_val
                improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
                row += f" {improvement:+.4f} ({improvement_pct:+.2f}%)"
            
            print(row)
        
        # 小目标类别对比
        if 'class_aps' in baseline:
            print("\n" + "-" * 70)
            print("小目标类别详细对比 (mAP50-95)")
            print("-" * 70)
            
            small_target_classes = ['pedestrian', 'people', 'bicycle', 'motor', 'tricycle', 'awning-tricycle']
            
            for cls in small_target_classes:
                if cls in baseline['class_aps']:
                    row = f"{cls:<20}"
                    baseline_ap = baseline['class_aps'][cls]
                    row += f" {baseline_ap:<18.4f}"
                    
                    best_improvement = 0
                    for other_model in other_models:
                        if 'class_aps' in other_model and cls in other_model['class_aps']:
                            other_ap = other_model['class_aps'][cls]
                            row += f" {other_ap:<18.4f}"
                            improvement = other_ap - baseline_ap
                            if improvement > best_improvement:
                                best_improvement = improvement
                        else:
                            row += f" {'N/A':<18}"
                    
                    if best_improvement != 0:
                        improvement_pct = (best_improvement / baseline_ap * 100) if baseline_ap > 0 else 0
                        row += f" {best_improvement:+.4f} ({improvement_pct:+.2f}%)"
                    
                    print(row)
        
        print("\n" + "=" * 70)
        print("结论:")
        baseline_map = baseline['map50_95']
        
        for i, other_model in enumerate(other_models):
            model_name = other_models[i]['model_name']
            improvement = other_model['map50_95'] - baseline_map
            
            if improvement > 0.01:
                print(f"✓ {model_name} 相比基线有显著提升 (+{improvement:.4f} mAP50-95, +{improvement/baseline_map*100:.1f}%)")
            elif improvement > 0:
                print(f"✓ {model_name} 相比基线有轻微提升 (+{improvement:.4f} mAP50-95, +{improvement/baseline_map*100:.1f}%)")
            elif improvement > -0.01:
                print(f"≈ {model_name} 相比基线基本持平 ({improvement:+.4f} mAP50-95)")
            else:
                print(f"⚠ {model_name} 相比基线下降了 ({improvement:.4f} mAP50-95, {improvement/baseline_map*100:.1f}%)")
        
        # 找出最佳模型
        if len(other_models) > 0:
            best_model = max(other_models, key=lambda x: x['map50_95'])
            print(f"\n🏆 最佳模型: {best_model['model_name']} (mAP50-95: {best_model['map50_95']:.4f})")
        
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

