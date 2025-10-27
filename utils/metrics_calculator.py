#!/usr/bin/env python3
"""
完整指标计算器
"""

import numpy as np
from pathlib import Path


class MetricsCalculator:
    """指标计算器 - 计算F1等额外指标"""

    def __init__(self):
        self.metrics = {}

    def calculate_f1(self, precision, recall):
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_f2(self, precision, recall, beta=2):
        """计算F2分数（更关注召回率）"""
        if precision + recall == 0:
            return 0.0
        return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)

    def calculate_ap_per_class(self, precision, recall):
        """计算各类别的平均精度（简化版）"""
        # 这里可以集成更复杂的AP计算逻辑
        return precision  # 简化处理

    def calculate_comprehensive_metrics(self, yolo_results):
        """计算综合指标"""

        # 基础指标
        precision = yolo_results.get('precision', 0)
        recall = yolo_results.get('recall', 0)
        map50 = yolo_results.get('map50', 0)
        map50_95 = yolo_results.get('map50_95', 0)

        # 计算衍生指标
        f1 = self.calculate_f1(precision, recall)
        f2 = self.calculate_f2(precision, recall)

        # 构建完整指标字典
        comprehensive_metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'f2_score': f2,
            'map50': map50,
            'map50_95': map50_95,
            'fps': yolo_results.get('fps', 0),
            'params': yolo_results.get('params', 0),
            'efficiency_score': self.calculate_efficiency_score(map50, yolo_results.get('fps', 0))
        }

        return comprehensive_metrics

    def calculate_efficiency_score(self, map50, fps, alpha=0.7):
        """计算效率得分（平衡精度和速度）"""
        # 归一化处理
        normalized_map = min(map50 / 0.5, 1.0)  # 假设0.5为参考值
        normalized_fps = min(fps / 60, 1.0)  # 假设60FPS为参考值

        return alpha * normalized_map + (1 - alpha) * normalized_fps

    def parse_yolo_results(self, results_dir):
        """解析YOLO训练结果"""
        # 这里需要根据YOLO的输出文件格式来解析
        # 暂时返回示例数据
        return {
            'precision': 0.467,
            'recall': 0.356,
            'map50': 0.421,
            'map50_95': 0.235,
            'fps': 45,
            'params': 11.2
        }


# 使用示例
def main():
    calculator = MetricsCalculator()

    # 示例数据
    yolo_results = {
        'precision': 0.467,
        'recall': 0.356,
        'map50': 0.421,
        'map50_95': 0.235,
        'fps': 45,
        'params': 11.2
    }

    comprehensive = calculator.calculate_comprehensive_metrics(yolo_results)

    print("📊 完整指标报告:")
    for metric, value in comprehensive.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()