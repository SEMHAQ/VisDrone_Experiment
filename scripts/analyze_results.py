#!/usr/bin/env python3
"""
完整结果分析脚本
"""

import os
import sys
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.path_manager import path_manager


class ComprehensiveAnalyzer:
    """综合分析器"""

    def __init__(self):
        self.results_data = {}
        self.experiments = ['baseline', 'ema_attention', 'bifpn', 'full_model']

    def collect_all_metrics(self):
        """收集所有实验的指标"""
        print("📊 收集实验结果数据...")

        for exp_name in self.experiments:
            exp_dir = path_manager.get_experiment_dir(exp_name)

            # 尝试从不同位置收集数据
            metrics = self._collect_metrics_from_files(exp_dir, exp_name)
            self.results_data[exp_name] = metrics

            print(f"   {exp_name}: {len(metrics)} 个指标收集完成")

    def _collect_metrics_from_files(self, exp_dir, exp_name):
        """从文件收集指标"""
        metrics = {
            'experiment': exp_name,
            'timestamp': datetime.now().isoformat()
        }

        # 1. 从评估结果文件读取
        eval_file = exp_dir / "metrics" / "evaluation_results.txt"
        if eval_file.exists():
            metrics.update(self._parse_evaluation_file(eval_file))

        # 2. 从训练日志读取
        train_log = exp_dir / "train_log.txt"
        if not train_log.exists():
            # 尝试查找其他日志文件
            for log_file in exp_dir.glob("*.log"):
                train_log = log_file
                break

        if train_log.exists():
            metrics.update(self._parse_training_log(train_log))

        # 3. 从YOLO结果文件读取
        results_file = exp_dir / "results.csv"
        if results_file.exists():
            metrics.update(self._parse_results_csv(results_file))

        # 4. 检查模型文件大小
        weights_file = exp_dir / "weights" / "best.pt"
        if weights_file.exists():
            metrics['model_size_mb'] = weights_file.stat().st_size / (1024 * 1024)

        return metrics

    def _parse_evaluation_file(self, file_path):
        """解析评估结果文件"""
        metrics = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()

                        # 转换为数值
                        try:
                            if '.' in value:
                                metrics[key] = float(value)
                            else:
                                metrics[key] = int(value)
                        except ValueError:
                            metrics[key] = value
        except Exception as e:
            print(f"⚠ 解析评估文件失败 {file_path}: {e}")

        return metrics

    def _parse_training_log(self, file_path):
        """解析训练日志"""
        metrics = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 简单的解析逻辑，实际应该更复杂
                for line in lines:
                    if 'epoch' in line.lower() and 'time' in line.lower():
                        # 提取训练时间信息
                        pass
        except Exception as e:
            print(f"⚠ 解析训练日志失败 {file_path}: {e}")

        return metrics

    def _parse_results_csv(self, file_path):
        """解析YOLO结果CSV"""
        metrics = {}
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                # 获取最后一行（最终结果）
                last_row = df.iloc[-1]
                metrics.update({
                    'train_loss': last_row.get('train/loss', 0),
                    'val_loss': last_row.get('val/loss', 0),
                    'precision': last_row.get('metrics/precision', 0),
                    'recall': last_row.get('metrics/recall', 0),
                    'map50': last_row.get('metrics/mAP50', 0),
                    'map50_95': last_row.get('metrics/mAP50-95', 0)
                })
        except Exception as e:
            print(f"⚠ 解析结果CSV失败 {file_path}: {e}")

        return metrics

    def create_comparison_table(self):
        """创建对比表格"""
        if not self.results_data:
            print("❌ 没有可用的结果数据")
            return None

        # 准备数据
        rows = []
        for exp_name, metrics in self.results_data.items():
            row = {
                '实验名称': exp_name,
                'mAP@0.5': metrics.get('map@0.5', metrics.get('map50', 0)),
                'mAP@0.5:0.95': metrics.get('map@0.5:0.95', metrics.get('map50_95', 0)),
                '精确率': metrics.get('precision', 0),
                '召回率': metrics.get('recall', 0),
                'F1分数': self._calculate_f1_score(metrics.get('precision', 0), metrics.get('recall', 0)),
                '模型大小(MB)': metrics.get('model_size_mb', 0),
                '状态': '已完成' if metrics.get('model_size_mb', 0) > 0 else '未完成'
            }
            rows.append(row)

        # 创建DataFrame
        df = pd.DataFrame(rows)

        # 保存为CSV
        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "experiment_comparison.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 对比表格已保存: {csv_path}")

        # 打印表格
        print("\n📋 实验对比结果:")
        print("=" * 80)
        print(df.to_string(index=False))

        return df

    def _calculate_f1_score(self, precision, recall):
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def generate_visualizations(self, df):
        """生成可视化图表"""
        if df is None or df.empty:
            print("❌ 没有数据可生成图表")
            return

        print("\n📈 生成可视化图表...")

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图表目录
        viz_dir = Path("results/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 1. 精度对比柱状图
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        sns.barplot(data=df, x='实验名称', y='mAP@0.5')
        plt.title('mAP@0.5 对比')
        plt.xticks(rotation=45)

        plt.subplot(2, 2, 2)
        sns.barplot(data=df, x='实验名称', y='mAP@0.5:0.95')
        plt.title('mAP@0.5:0.95 对比')
        plt.xticks(rotation=45)

        plt.subplot(2, 2, 3)
        sns.barplot(data=df, x='实验名称', y='精确率')
        plt.title('精确率对比')
        plt.xticks(rotation=45)

        plt.subplot(2, 2, 4)
        sns.barplot(data=df, x='实验名称', y='召回率')
        plt.title('召回率对比')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(viz_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 综合指标雷达图
        self._create_radar_chart(df, viz_dir)

        print(f"✅ 可视化图表已保存: {viz_dir}")

    def _create_radar_chart(self, df, viz_dir):
        """创建雷达图"""
        try:
            # 选择数值型列
            numeric_cols = ['mAP@0.5', 'mAP@0.5:0.95', '精确率', '召回率', 'F1分数']
            plot_data = df[['实验名称'] + numeric_cols].copy()

            # 归一化数据
            for col in numeric_cols:
                if plot_data[col].max() > 0:
                    plot_data[col] = plot_data[col] / plot_data[col].max()

            # 创建雷达图
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)

            # 设置角度
            angles = [n / float(len(numeric_cols)) * 2 * np.pi for n in range(len(numeric_cols))]
            angles += angles[:1]  # 闭合

            # 绘制每个实验
            for idx, row in plot_data.iterrows():
                values = row[numeric_cols].tolist()
                values += values[:1]  # 闭合
                ax.plot(angles, values, 'o-', linewidth=2, label=row['实验名称'])
                ax.fill(angles, values, alpha=0.1)

            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(numeric_cols)
            ax.set_ylim(0, 1)
            plt.title('实验性能雷达图', size=14, y=1.08)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

            plt.savefig(viz_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"⚠ 创建雷达图失败: {e}")

    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n📄 生成综合分析报告...")

        report_dir = Path("results")
        report_dir.mkdir(parents=True, exist_ok=True)

        # 创建报告
        report_content = self._build_report_content()

        # 保存Markdown报告
        md_report = report_dir / "comprehensive_analysis_report.md"
        with open(md_report, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # 保存JSON数据
        json_data = report_dir / "experiment_results.json"
        with open(json_data, 'w', encoding='utf-8') as f:
            json.dump(self.results_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 分析报告已生成:")
        print(f"   📝 Markdown报告: {md_report}")
        print(f"   📊 JSON数据: {json_data}")

        return md_report

    def _build_report_content(self):
        """构建报告内容"""
        content = "# VisDrone实验综合分析报告\n\n"
        content += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        content += "## 实验概览\n\n"
        content += "| 实验 | 状态 | mAP@0.5 | mAP@0.5:0.95 | 精确率 | 召回率 | F1分数 |\n"
        content += "|------|------|---------|--------------|--------|--------|--------|\n"

        for exp_name, metrics in self.results_data.items():
            content += f"| {exp_name} | "
            content += "✅ 已完成 | " if metrics.get('model_size_mb', 0) > 0 else "⏳ 进行中 | "
            content += f"{metrics.get('map@0.5', metrics.get('map50', 0)):.4f} | "
            content += f"{metrics.get('map@0.5:0.95', metrics.get('map50_95', 0)):.4f} | "
            content += f"{metrics.get('precision', 0):.4f} | "
            content += f"{metrics.get('recall', 0):.4f} | "
            content += f"{self._calculate_f1_score(metrics.get('precision', 0), metrics.get('recall', 0)):.4f} |\n"

        content += "\n## 详细分析\n\n"

        # 性能分析
        content += "### 性能分析\n\n"
        completed_exps = {k: v for k, v in self.results_data.items() if v.get('model_size_mb', 0) > 0}

        if completed_exps:
            best_map50 = max(metrics.get('map@0.5', metrics.get('map50', 0)) for metrics in completed_exps.values())
            best_exp = [k for k, v in completed_exps.items()
                        if v.get('map@0.5', v.get('map50', 0)) == best_map50][0]

            content += f"- **最佳性能模型**: {best_exp} (mAP@0.5: {best_map50:.4f})\n"
            content += f"- **相对基准提升**: {((best_map50 - completed_exps['baseline'].get('map@0.5', completed_exps['baseline'].get('map50', 0))) / completed_exps['baseline'].get('map@0.5', completed_exps['baseline'].get('map50', 0)) * 100):.2f}%\n\n"

        # 改进效果分析
        content += "### 改进效果分析\n\n"
        content += "1. **基准模型**: 作为性能基准\n"
        content += "2. **+EMA注意力**: 关注特征增强效果\n"
        content += "3. **+BiFPN**: 关注多尺度特征融合效果\n"
        content += "4. **完整模型**: 综合改进效果\n\n"

        content += "## 结论与建议\n\n"
        content += "基于当前实验结果，建议：\n\n"
        content += "1. 继续完成所有实验的训练\n"
        content += "2. 对表现最佳的改进模块进行深入分析\n"
        content += "3. 考虑模型复杂度和推理速度的平衡\n"

        return content


def run_analysis():
    """运行结果分析"""
    analyzer = ComprehensiveAnalyzer()

    # 收集数据
    analyzer.collect_all_metrics()

    # 生成对比表格
    df = analyzer.create_comparison_table()

    # 生成可视化
    analyzer.generate_visualizations(df)

    # 生成报告
    report_path = analyzer.generate_comprehensive_report()

    print(f"\n🎉 结果分析完成!")
    print(f"📁 所有结果保存在: results/ 目录")

    return report_path


if __name__ == "__main__":
    # 确保numpy可用
    try:
        import numpy as np
    except ImportError:
        print("❌ 需要安装numpy: pip install numpy")
        sys.exit(1)

    run_analysis()