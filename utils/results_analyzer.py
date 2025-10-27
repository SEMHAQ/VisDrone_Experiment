#!/usr/bin/env python3
"""
结果分析工具
"""

import json
import yaml
from pathlib import Path
import pandas as pd
from .path_manager import path_manager


class ResultsAnalyzer:
    """结果分析器"""

    def __init__(self):
        self.results_data = {}

    def collect_all_results(self):
        """收集所有实验结果"""
        experiments = ["baseline", "image_enhance", "ema_attention", "bifpn", "full_model"]

        for exp_name in experiments:
            exp_dir = path_manager.get_experiment_dir(exp_name)
            metrics_file = exp_dir / "metrics" / "evaluation_results.txt"

            if metrics_file.exists():
                self.results_data[exp_name] = self.parse_results_file(metrics_file)
            else:
                print(f"⚠  {exp_name}: 结果文件不存在")

    def parse_results_file(self, file_path):
        """解析结果文件"""
        results = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()

                        # 尝试转换为数值
                        try:
                            if '.' in value:
                                results[key] = float(value)
                            else:
                                results[key] = int(value)
                        except ValueError:
                            results[key] = value
        except Exception as e:
            print(f"❌ 解析文件失败 {file_path}: {e}")

        return results

    def create_comparison_table(self):
        """创建对比表格"""
        if not self.results_data:
            print("❌ 没有可用的结果数据")
            return

        # 创建DataFrame
        df_data = []
        for exp_name, results in self.results_data.items():
            row = {
                '实验': exp_name,
                'mAP@0.5': results.get('mAP@0.5', 0),
                'mAP@0.5:0.95': results.get('mAP@0.5:0.95', 0),
                '精确率': results.get('精确率', 0),
                '召回率': results.get('召回率', 0)
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # 保存为CSV
        output_file = path_manager.project_root / "results" / "experiment_comparison.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 对比表格已保存: {output_file}")

        # 打印表格
        print("\n📊 实验对比结果:")
        print("=" * 60)
        print(df.to_string(index=False))

        return df

    def generate_report(self):
        """生成分析报告"""
        if not self.results_data:
            print("❌ 没有可用的结果数据")
            return

        report_file = path_manager.project_root / "results" / "analysis_report.md"
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# VisDrone实验分析报告\n\n")

            f.write("## 实验结果概览\n\n")
            f.write("| 实验 | mAP@0.5 | mAP@0.5:0.95 | 精确率 | 召回率 |\n")
            f.write("|------|---------|--------------|--------|--------|\n")

            for exp_name, results in self.results_data.items():
                f.write(
                    f"| {exp_name} | {results.get('mAP@0.5', 0):.4f} | {results.get('mAP@0.5:0.95', 0):.4f} | {results.get('精确率', 0):.4f} | {results.get('召回率', 0):.4f} |\n")

            f.write("\n## 分析结论\n\n")
            # 这里可以添加自动分析逻辑
            f.write("实验数据分析待完善...\n")

        print(f"✅ 分析报告已生成: {report_file}")


def analyze_results():
    """分析所有结果"""
    analyzer = ResultsAnalyzer()
    analyzer.collect_all_results()
    analyzer.create_comparison_table()
    analyzer.generate_report()


if __name__ == "__main__":
    analyze_results()