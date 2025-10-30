#!/usr/bin/env python3
"""
修复的综合评估所有实验模型
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO
from utils.path_manager import path_manager


class FixedComprehensiveEvaluator:
    """修复的综合评估器"""

    def __init__(self):
        self.experiments = {
            'baseline': {
                'name': 'baseline',
                'description': '原始YOLOv8s',
                'weights_path': path_manager.get_experiment_dir('baseline') / 'weights' / 'best.pt'
            },
            'ema': {
                'name': 'ema',
                'description': 'YOLOv8s + EMA注意力',
                'weights_path': path_manager.get_experiment_dir('ema') / 'weights' / 'best.pt'
            },
            'bifpn': {
                'name': 'bifpn',
                'description': 'YOLOv8s + BiFPN',
                'weights_path': path_manager.get_experiment_dir('bifpn') / 'weights' / 'best.pt'
            },
            'full': {
                'name': 'full',
                'description': 'YOLOv8s + EMA + BiFPN',
                'weights_path': path_manager.get_experiment_dir('full') / 'weights' / 'best.pt'
            }
        }

        self.results = {}

    def check_model_files(self):
        """检查模型文件是否存在"""
        print("🔍 检查模型文件...")
        missing_models = []

        for exp_name, config in self.experiments.items():
            weights_path = config['weights_path']
            if weights_path.exists():
                print(f"✅ {exp_name}: {weights_path}")
            else:
                print(f"❌ {exp_name}: 模型文件不存在")
                missing_models.append(exp_name)

        return missing_models

    def extract_metrics_from_output(self, output_text):
        """从评估输出文本中提取指标"""
        # 解析评估输出文本，提取关键指标
        lines = output_text.split('\n')
        metrics = {}

        for line in lines:
            if 'all' in line and 'Images' in line:
                # 解析类似: "all        548      38759      0.517      0.395      0.402      0.237"
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        metrics['precision'] = float(parts[3])
                        metrics['recall'] = float(parts[4])
                        metrics['map50'] = float(parts[5])
                        metrics['map'] = float(parts[6])
                    except:
                        pass
            elif 'Speed:' in line:
                # 解析推理速度
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'inference' in part and i > 0:
                        try:
                            metrics['inference_speed'] = float(parts[i - 1].replace('ms', ''))
                        except:
                            pass

        return metrics

    def evaluate_single_model(self, exp_name, config):
        """评估单个模型 - 兼容版本"""
        print(f"\n📊 评估模型: {exp_name}")
        print(f"描述: {config['description']}")
        print("-" * 50)

        weights_path = config['weights_path']

        if not weights_path.exists():
            print(f"❌ 模型文件不存在: {weights_path}")
            return None

        try:
            # 加载模型
            model = YOLO(str(weights_path))

            # 在验证集上评估
            print("开始评估...")

            # 使用更兼容的方式获取指标
            import io
            import contextlib

            # 捕获评估输出
            output_capture = io.StringIO()
            with contextlib.redirect_stdout(output_capture):
                metrics = model.val(
                    data=str(path_manager.dataset_config),
                    split='val',
                    imgsz=640,
                    batch=16,
                    conf=0.001,
                    iou=0.6,
                    device='cpu',
                    verbose=True,  # 设置为True以获取详细输出
                    save_json=False  # 暂时关闭以避免干扰
                )

            output_text = output_capture.getvalue()

            # 从输出文本中提取指标
            extracted_metrics = self.extract_metrics_from_output(output_text)

            # 尝试使用新的API
            try:
                # 使用新的属性名称
                precision = metrics.box.p.mean() if hasattr(metrics.box, 'p') else extracted_metrics.get('precision', 0)
                recall = metrics.box.r.mean() if hasattr(metrics.box, 'r') else extracted_metrics.get('recall', 0)
                map50 = metrics.box.map50 if hasattr(metrics.box, 'map50') else extracted_metrics.get('map50', 0)
                map_val = metrics.box.map if hasattr(metrics.box, 'map') else extracted_metrics.get('map', 0)
            except:
                # 如果新API失败，使用提取的指标
                precision = extracted_metrics.get('precision', 0)
                recall = extracted_metrics.get('recall', 0)
                map50 = extracted_metrics.get('map50', 0)
                map_val = extracted_metrics.get('map', 0)

            result = {
                'map50': float(map50),
                'map': float(map_val),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': self.calculate_f1_score(float(precision), float(recall)),
                'inference_speed': extracted_metrics.get('inference_speed', 0)
            }

            print(f"✅ {exp_name} 评估完成:")
            print(f"   mAP@0.5:     {result['map50']:.4f}")
            print(f"   mAP@0.5:0.95: {result['map']:.4f}")
            print(f"   精确率:      {result['precision']:.4f}")
            print(f"   召回率:      {result['recall']:.4f}")
            print(f"   F1分数:      {result['f1_score']:.4f}")

            return result

        except Exception as e:
            print(f"❌ {exp_name} 评估失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_f1_score(self, precision, recall):
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate_all_models(self):
        """评估所有模型"""
        print("🧪 开始综合评估所有模型")
        print("=" * 60)

        # 检查模型文件
        missing_models = self.check_model_files()
        if missing_models:
            print(f"\n⚠ 以下模型文件缺失: {missing_models}")
            print("请先完成这些模型的训练")
            return False

        # 逐个评估模型
        for exp_name, config in self.experiments.items():
            result = self.evaluate_single_model(exp_name, config)
            self.results[exp_name] = result

        # 分析结果
        self.analyze_results()

        # 生成报告
        self.generate_comprehensive_report()

        return True

    def analyze_results(self):
        """分析评估结果"""
        print("\n📈 结果分析")
        print("=" * 60)

        # 过滤掉评估失败的模型
        valid_results = {k: v for k, v in self.results.items() if v is not None}

        if not valid_results:
            print("❌ 没有有效的评估结果")
            return

        # 创建性能对比表格
        df_data = []
        for exp_name, metrics in valid_results.items():
            row = {
                '实验': exp_name,
                '描述': self.experiments[exp_name]['description'],
                'mAP@0.5': metrics['map50'],
                'mAP@0.5:0.95': metrics['map'],
                '精确率': metrics['precision'],
                '召回率': metrics['recall'],
                'F1分数': metrics['f1_score'],
                '推理速度(ms/img)': metrics['inference_speed']
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # 显示性能对比
        print("\n📊 性能对比表:")
        print("=" * 80)
        print(df.to_string(index=False, float_format='%.4f'))

        # 计算改进百分比（相对于基准模型）
        if 'baseline' in valid_results:
            baseline_map50 = valid_results['baseline']['map50']
            baseline_map = valid_results['baseline']['map']
            baseline_f1 = valid_results['baseline']['f1_score']

            print(f"\n📈 相对于基准模型的改进:")
            print("=" * 50)
            print("| 实验 | mAP@0.5改进 | mAP改进 | F1改进 |")
            print("|------|------------|---------|--------|")

            for exp_name, metrics in valid_results.items():
                if exp_name != 'baseline':
                    map50_improvement = ((metrics['map50'] - baseline_map50) / baseline_map50) * 100
                    map_improvement = ((metrics['map'] - baseline_map) / baseline_map) * 100
                    f1_improvement = ((metrics['f1_score'] - baseline_f1) / baseline_f1) * 100

                    print(
                        f"| {exp_name} | {map50_improvement:+.2f}% | {map_improvement:+.2f}% | {f1_improvement:+.2f}% |")

        self.performance_df = df
        return df

    def generate_comprehensive_report(self):
        """生成综合评估报告"""
        print("\n📄 生成综合评估报告...")

        # 创建结果目录
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON格式的详细结果
        json_path = results_dir / "detailed_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✅ 详细结果已保存: {json_path}")

        # 保存CSV格式的性能对比
        if hasattr(self, 'performance_df'):
            csv_path = results_dir / "performance_comparison.csv"
            self.performance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ 性能对比表已保存: {csv_path}")

        # 生成Markdown报告
        self.generate_markdown_report(results_dir)

        print("🎉 综合评估完成!")

    def generate_markdown_report(self, results_dir):
        """生成Markdown格式的报告"""
        md_path = results_dir / "comprehensive_evaluation_report.md"

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# VisDrone消融实验综合评估报告\n\n")
            f.write("## 实验概述\n\n")

            # 实验列表
            f.write("| 实验名称 | 描述 | 状态 |\n")
            f.write("|----------|------|------|\n")
            for exp_name, config in self.experiments.items():
                status = "✅ 完成" if self.results.get(exp_name) else "❌ 失败"
                f.write(f"| {exp_name} | {config['description']} | {status} |\n")

            f.write("\n## 性能对比\n\n")

            if hasattr(self, 'performance_df'):
                f.write("| 实验 | mAP@0.5 | mAP@0.5:0.95 | 精确率 | 召回率 | F1分数 | 推理速度(ms/img) |\n")
                f.write("|------|---------|--------------|--------|--------|--------|----------------|\n")

                for _, row in self.performance_df.iterrows():
                    f.write(f"| {row['实验']} | {row['mAP@0.5']:.4f} | {row['mAP@0.5:0.95']:.4f} | "
                            f"{row['精确率']:.4f} | {row['召回率']:.4f} | {row['F1分数']:.4f} | {row['推理速度(ms/img)']:.2f} |\n")

            # 改进分析
            f.write("\n## 改进效果分析\n\n")

            baseline_result = self.results.get('baseline')
            if baseline_result:
                f.write("### 相对于基准模型的改进百分比\n\n")
                f.write("| 实验 | mAP@0.5改进 | mAP@0.5:0.95改进 | F1分数改进 |\n")
                f.write("|------|------------|-----------------|-----------|\n")

                for exp_name, result in self.results.items():
                    if exp_name != 'baseline' and result:
                        map50_improvement = ((result['map50'] - baseline_result['map50']) / baseline_result[
                            'map50']) * 100
                        map_improvement = ((result['map'] - baseline_result['map']) / baseline_result['map']) * 100
                        f1_improvement = ((result['f1_score'] - baseline_result['f1_score']) / baseline_result[
                            'f1_score']) * 100

                        f.write(
                            f"| {exp_name} | {map50_improvement:+.2f}% | {map_improvement:+.2f}% | {f1_improvement:+.2f}% |\n")

            # 结论
            f.write("\n## 结论\n\n")

            # 自动生成结论
            best_model = None
            best_map50 = 0

            for exp_name, result in self.results.items():
                if result and result['map50'] > best_map50:
                    best_map50 = result['map50']
                    best_model = exp_name

            if best_model and baseline_result:
                improvement = ((best_map50 - baseline_result['map50']) / baseline_result['map50']) * 100
                f.write(f"- **最佳模型**: {best_model} (mAP@0.5: {best_map50:.4f})\n")
                f.write(f"- **相对于基准模型提升**: {improvement:+.2f}%\n")

            f.write("\n## 建议\n\n")
            f.write("1. 根据实验结果选择最适合的模型进行部署\n")
            f.write("2. 考虑模型复杂度和推理速度的平衡\n")
            f.write("3. 进一步优化表现最佳的改进模块\n")

        print(f"✅ Markdown报告已生成: {md_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("       修复的VisDrone消融实验综合评估")
    print("=" * 60)

    # 验证环境
    if not path_manager.validate_paths():
        print("❌ 环境验证失败")
        return

    evaluator = FixedComprehensiveEvaluator()
    success = evaluator.evaluate_all_models()

    if success:
        print("\n🎯 评估完成!")
        print("📁 所有结果保存在 results/ 目录")
    else:
        print("\n❌ 评估失败")


if __name__ == "__main__":
    main()