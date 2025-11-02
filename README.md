# VisDrone 目标检测实验项目

基于 YOLOv8s 的 VisDrone 航拍小目标检测改进实验

---

## 📊 项目概述

### 实验目标
- **基线**: YOLOv8s mAP50-95 = 34.49%
- **目标**: 提升到 **42-45%** (+22-30%)
- **重点**: 小目标类别（bicycle, people, pedestrian）提升 **40-70%**

### 硬件环境
- **GPU**: NVIDIA RTX 3060 12GB
- **图像分辨率**: 1024×1024
- **训练轮数**: 200-300 epochs

### 数据集
- **训练集**: 6,471 张图片
- **验证集**: 548 张图片
- **类别数**: 10 类

---

## 📁 项目结构

```
VisDrone_Experiment/
├── docs/                          # 📚 文档目录
│   ├── TRAINING_GUIDE.md          # 训练指南（推荐阅读）
│   ├── EXPERIMENT_SCHEDULE.md     # 详细实验日程
│   ├── QUICK_REFERENCE.md         # 快速参考
│   └── README_EXPERIMENTS.md      # 实验总览
│
├── scripts/                       # 🔧 脚本目录
│   ├── train/                     # 训练脚本
│   │   ├── train_baseline.py
│   │   ├── train_p2.py
│   │   ├── train_p2_bifpn.py
│   │   ├── train_p2_bifpn_dcn.py
│   │   ├── train_p2_bifpn_carafe.py
│   │   └── train_final_improved.py
│   │
│   ├── eval/                      # 评估脚本
│   │   ├── eval_model.py
│   │   ├── compare_results.py
│   │   └── check_data.py
│   │
│   └── inference/                 # 推理脚本
│       ├── test_inference.py
│       └── test_ultralytics.py
│
├── cfg/                           # ⚙️ 配置文件
│   ├── models/                    # 模型配置
│   ├── visdrone.yaml              # 数据集配置
│   └── train_base1024.yaml        # 训练配置
│
├── VisDrone2YOLO/                 # 📊 数据集
│
├── results/                       # 📈 结果目录
│   ├── expected/                  # 预期结果文档
│   └── ...
│
├── runs/                          # 🏃 训练输出
│
├── weights/                       # 💾 预训练权重
│
└── README.md                      # 📖 本文件
```

---

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install ultralytics

# 检查数据完整性
python scripts/eval/check_data.py
```

### 2. 训练模型

```bash
# Baseline（已完成）
python scripts/train/train_baseline.py

# P2模型（需重训练）
python scripts/train/train_p2.py

# P2 + BiFPN
python scripts/train/train_p2_bifpn.py

# 最终改进模型
python scripts/train/train_final_improved.py
```

### 3. 评估模型

```bash
# 单模型评估
python scripts/eval/eval_model.py <模型权重路径>

# 对比所有模型
python scripts/eval/compare_results.py
```

---

## 📊 实验结果

### 当前进度
- ✅ **Baseline**: mAP50-95 = 34.49%
- ⚠️ **P2**: 需重训练（配置已修复）
- ⏳ **后续实验**: 待执行

### 预期结果汇总

| 模型 | mAP50-95 | bicycle | people | 训练时间 | 状态 |
|------|----------|---------|--------|----------|------|
| Baseline | 34.49% | 16.84% | 24.57% | - | ✅ |
| P2 | 38-40% | 22-28% | 29-35% | 20h | ⚠️ |
| P2+BiFPN | 40-43% | 25-32% | 32-38% | 22h | ⏳ |
| P2+BiFPN+DCN | 41-44% | 26-33% | 33-39% | 24h | ⏳ |
| P2+BiFPN+CARAFE | 41-44% | 26-33% | 33-39% | 22h | ⏳ |
| **最终模型** | **42-45%** | **28-35%** | **35-42%** | 24h | ⏳ |

详细预期结果请查看 `results/expected/` 目录

---

## 📚 文档导航

### 新手入门
1. **阅读**: `docs/TRAINING_GUIDE.md` - 训练指南
2. **参考**: `docs/QUICK_REFERENCE.md` - 快速参考
3. **查看**: `results/expected/` - 预期结果

### 详细文档
- `docs/EXPERIMENT_SCHEDULE.md` - 完整实验日程（12步）
- `docs/README_EXPERIMENTS.md` - 实验总览

---

## 🎯 推荐实验路线

### 核心实验（推荐）
1. ✅ Baseline - 已完成
2. ⏳ P2 模型 → 预期 38-40%
3. ⏳ P2 + BiFPN → 预期 40-43%
4. ⏳ 最终改进模型 → 预期 42-45%

**总时间**: ~66小时（约3天）

---

## 🔧 常用命令

### 训练
```bash
# P2模型（当前任务）
python scripts/train/train_p2.py

# BiFPN模型（下一步）
python scripts/train/train_p2_bifpn.py
```

### 评估
```bash
# 单模型评估
python scripts/eval/eval_model.py <model_path>

# 对比所有模型
python scripts/eval/compare_results.py
```

### 监控
```bash
# GPU监控
nvidia-smi -l 1
```

---

## ⚙️ 配置说明

### 12GB RTX 3060 显存优化

| 模型 | 推荐Batch | 最小Batch | 显存占用 |
|------|-----------|-----------|----------|
| Baseline | 6 | 4 | 10GB |
| P2 | 4 | 3 | 11GB |
| P2+BiFPN | 4 | 3 | 11-12GB |
| P2+BiFPN+DCN | 3-4 | 2 | 11-12GB |

**OOM解决方案**:
1. 降低batch: 4→3→2
2. 梯度累积: `accumulate=2`
3. 降低分辨率: 1024→960

---

## 📝 注意事项

1. **训练时间**: 每个模型训练需要 20-24 小时
2. **显存占用**: 12GB RTX 3060 适配 batch=4
3. **对比公平性**: 所有实验使用相同的训练配置（除改进点外）
4. **结果记录**: 每次实验后运行评估脚本记录结果

---

**最后更新**: 2025-11-02  
**当前状态**: 项目结构已重组，P2模型待重训练

