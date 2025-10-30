#!/usr/bin/env python3
"""
模型诊断脚本 - 检查改进是否真正生效
"""

import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO


def diagnose_model(model_path, model_name):
    """诊断模型是否包含改进"""
    print(f"\n🔍 诊断模型: {model_name}")
    print("=" * 50)

    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return

    try:
        # 加载模型
        model = YOLO(str(model_path))

        # 检查模型结构
        print("📋 模型结构分析:")
        print(f"  模型类型: {type(model.model)}")
        print(f"  参数量: {sum(p.numel() for p in model.model.parameters()):,}")

        # 检查是否有EMA模块
        ema_modules = []
        for name, module in model.model.named_modules():
            if 'ema' in name.lower() or 'attention' in name.lower():
                ema_modules.append(name)

        if ema_modules:
            print(f"✅ 找到EMA/注意力模块: {ema_modules}")
        else:
            print("❌ 未找到EMA/注意力模块")

        # 检查是否有BiFPN模块
        bifpn_modules = []
        for name, module in model.model.named_modules():
            if 'bifpn' in name.lower() or 'fpn' in name.lower():
                bifpn_modules.append(name)

        if bifpn_modules:
            print(f"✅ 找到BiFPN模块: {bifpn_modules}")
        else:
            print("❌ 未找到BiFPN模块")

        # 检查模型参数是否不同
        print("\n📊 模型参数对比:")
        baseline_params = None

        if model_name == 'baseline':
            # 保存基准模型参数作为参考
            baseline_params = [p.clone() for p in model.model.parameters()]
            print("✅ 保存基准模型参数作为参考")
        else:
            # 加载基准模型进行比较
            baseline_path = Path("runs/baseline/weights/best.pt")
            if baseline_path.exists():
                baseline_model = YOLO(str(baseline_path))
                baseline_params = [p for p in baseline_model.model.parameters()]

                # 比较参数差异
                current_params = [p for p in model.model.parameters()]
                param_diffs = []

                for i, (base_param, current_param) in enumerate(zip(baseline_params, current_params)):
                    diff = torch.mean(torch.abs(base_param - current_param)).item()
                    param_diffs.append(diff)

                avg_diff = sum(param_diffs) / len(param_diffs)
                max_diff = max(param_diffs)

                print(f"  平均参数差异: {avg_diff:.6f}")
                print(f"  最大参数差异: {max_diff:.6f}")

                if avg_diff < 1e-6:
                    print("⚠ 参数差异极小，可能是相同的模型")
                else:
                    print("✅ 参数存在显著差异")

        return True

    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        return False


def main():
    """主诊断函数"""
    print("=" * 60)
    print("       模型改进诊断工具")
    print("=" * 60)

    models = {
        'baseline': Path("runs/baseline/weights/best.pt"),
        'ema': Path("runs/ema/weights/best.pt"),
        'bifpn': Path("runs/bifpn/weights/best.pt"),
        'full': Path("runs/full/weights/best.pt")
    }

    # 检查所有模型
    for name, path in models.items():
        diagnose_model(path, name)

    print("\n" + "=" * 60)
    print("诊断完成!")
    print("如果所有模型的参数差异极小，说明改进模块没有正确集成")


if __name__ == "__main__":
    main()