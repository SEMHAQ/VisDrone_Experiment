# 项目结构说明

本文档详细说明了项目的目录结构和文件组织方式。

---

## 📁 完整目录结构

```
VisDrone_Experiment/
│
├── docs/                                    # 📚 文档目录
│   ├── TRAINING_GUIDE.md                    # 训练指南（推荐新手阅读）
│   ├── EXPERIMENT_SCHEDULE.md               # 详细实验日程（12步完整版）
│   ├── QUICK_REFERENCE.md                   # 快速参考卡片
│   ├── README_EXPERIMENTS.md                # 实验总览
│   └── PROJECT_STRUCTURE.md                 # 本文件
│
├── scripts/                                 # 🔧 脚本目录
│   │
│   ├── train/                               # 训练脚本
│   │   ├── train_baseline.py                # Baseline训练（YOLOv8s）
│   │   ├── train_p2.py                      # P2模型训练
│   │   ├── train_p2_bifpn.py                # P2+BiFPN训练
│   │   ├── train_p2_bifpn_dcn.py            # P2+BiFPN+DCN训练
│   │   ├── train_p2_bifpn_carafe.py         # P2+BiFPN+CARAFE训练
│   │   ├── train_final_improved.py          # 最终改进模型训练
│   │   └── train_p2_improved.py             # P2改进版（备用）
│   │
│   ├── eval/                                # 评估脚本
│   │   ├── eval_model.py                    # 单模型评估
│   │   ├── compare_results.py               # 多模型对比
│   │   └── check_data.py                    # 数据完整性检查
│   │
│   └── inference/                           # 推理脚本
│       ├── test_inference.py                # 推理测试
│       └── test_ultralytics.py              # Ultralytics功能测试
│
├── cfg/                                     # ⚙️ 配置文件
│   │
│   ├── models/                              # 模型配置文件
│   │   ├── yolov8s-p2.yaml                  # P2模型配置
│   │   ├── yolov8s-p2-bifpn.yaml            # P2+BiFPN配置
│   │   ├── yolov8s-p2-bifpn-dcn.yaml        # P2+BiFPN+DCN配置
│   │   ├── yolov8s-p2-bifpn-carafe.yaml     # P2+BiFPN+CARAFE配置
│   │   └── yolov8s-p2-bifpn-final.yaml      # 最终模型配置
│   │
│   ├── visdrone.yaml                        # 数据集配置
│   └── train_base1024.yaml                  # 训练基础配置
│
├── VisDrone2YOLO/                           # 📊 数据集目录
│   ├── VisDrone2019-DET-train/              # 训练集
│   │   ├── images/                          # 图片（6,471张）
│   │   └── labels/                          # 标签
│   │
│   ├── VisDrone2019-DET-val/                # 验证集
│   │   ├── images/                          # 图片（548张）
│   │   └── labels/                          # 标签
│   │
│   ├── VisDrone2019-DET-test-dev/           # 测试集（可选）
│   │   ├── images/
│   │   └── labels/
│   │
│   ├── visDrone2YOLO.py                     # 数据转换脚本
│   ├── filterVisDroneLabels.py              # 标签过滤脚本
│   ├── viewConvertedLabels.py               # 标签可视化脚本
│   └── README.md                            # 数据集说明
│
├── results/                                 # 📈 结果目录
│   │
│   ├── expected/                            # 预期结果文档
│   │   ├── p2_expected_results.md           # P2模型预期结果
│   │   ├── bifpn_expected_results.md        # BiFPN模型预期结果
│   │   ├── dcn_expected_results.md          # DCN模型预期结果
│   │   ├── carafe_expected_results.md       # CARAFE模型预期结果
│   │   └── final_expected_results.md        # 最终模型预期结果
│   │
│   ├── baseline_val/                        # Baseline评估结果
│   │   ├── confusion_matrix.png
│   │   ├── PR_curve.png
│   │   ├── F1_curve.png
│   │   └── ...
│   │
│   ├── baseline_val_log/                    # Baseline验证日志
│   ├── baseline_summary.md                  # Baseline结果总结
│   │
│   ├── p2_val/                              # P2评估结果
│   ├── p2_val_log/                          # P2验证日志
│   ├── p2_analysis.md                       # P2结果分析
│   └── p2_comparison_summary.md             # P2对比总结
│
├── runs/                                    # 🏃 训练输出目录
│   └── visdrone/                            # VisDrone实验输出
│       ├── baseline_y8s_1024_adamw/         # Baseline训练输出
│       ├── y8s_p2_1024_adamw_300ep/         # P2训练输出
│       └── ...                              # 其他实验输出
│
├── weights/                                 # 💾 预训练权重
│   ├── yolov8s.pt                           # YOLOv8s预训练权重
│   └── yolo11n.pt                           # YOLO11n预训练权重
│
└── README.md                                # 📖 项目主README
```

---

## 📂 目录说明

### 1. `docs/` - 文档目录

存放所有项目文档，与代码分离。

| 文件 | 说明 | 推荐阅读顺序 |
|------|------|--------------|
| `TRAINING_GUIDE.md` | 训练指南，包含所有实验的详细说明 | ⭐⭐⭐ 第一个阅读 |
| `QUICK_REFERENCE.md` | 快速参考卡片，常用命令和配置 | ⭐⭐ 随时查阅 |
| `EXPERIMENT_SCHEDULE.md` | 完整的12步实验日程 | ⭐ 详细规划时阅读 |
| `README_EXPERIMENTS.md` | 实验总览 | ⭐ 了解全局时阅读 |
| `PROJECT_STRUCTURE.md` | 项目结构说明（本文件） | 熟悉项目时阅读 |

---

### 2. `scripts/` - 脚本目录

所有可执行脚本，按功能分类。

#### 2.1 `scripts/train/` - 训练脚本

| 脚本 | 说明 | 预期结果 | 状态 |
|------|------|----------|------|
| `train_baseline.py` | Baseline训练 | 34.49% | ✅ 已完成 |
| `train_p2.py` | P2模型训练 | 38-40% | ⚠️ 需重训练 |
| `train_p2_bifpn.py` | P2+BiFPN训练 | 40-43% | ⏳ 待执行 |
| `train_p2_bifpn_dcn.py` | P2+BiFPN+DCN训练 | 41-44% | ⏳ 可选 |
| `train_p2_bifpn_carafe.py` | P2+BiFPN+CARAFE训练 | 41-44% | ⏳ 可选 |
| `train_final_improved.py` | 最终模型训练 | 42-45% | ⏳ 待执行 |

**使用方式**:
```bash
# 从项目根目录运行
python scripts/train/train_p2.py
```

#### 2.2 `scripts/eval/` - 评估脚本

| 脚本 | 说明 | 用途 |
|------|------|------|
| `eval_model.py` | 单模型评估 | 评估单个模型的性能 |
| `compare_results.py` | 多模型对比 | 对比所有模型的性能 |
| `check_data.py` | 数据完整性检查 | 检查数据集是否完整 |

**使用方式**:
```bash
# 评估单个模型
python scripts/eval/eval_model.py runs/visdrone/xxx/weights/best.pt

# 对比所有模型
python scripts/eval/compare_results.py

# 检查数据
python scripts/eval/check_data.py
```

#### 2.3 `scripts/inference/` - 推理脚本

| 脚本 | 说明 |
|------|------|
| `test_inference.py` | 推理测试 |
| `test_ultralytics.py` | Ultralytics功能测试 |

---

### 3. `cfg/` - 配置文件目录

#### 3.1 `cfg/models/` - 模型配置

| 配置文件 | 说明 |
|----------|------|
| `yolov8s-p2.yaml` | P2模型：添加stride=4检测头 |
| `yolov8s-p2-bifpn.yaml` | P2+BiFPN：双向特征金字塔 |
| `yolov8s-p2-bifpn-dcn.yaml` | P2+BiFPN+DCN：可变形卷积 |
| `yolov8s-p2-bifpn-carafe.yaml` | P2+BiFPN+CARAFE：内容感知上采样 |
| `yolov8s-p2-bifpn-final.yaml` | 最终模型：综合改进 |

#### 3.2 其他配置

| 配置文件 | 说明 |
|----------|------|
| `visdrone.yaml` | 数据集配置（路径、类别等） |
| `train_base1024.yaml` | 训练基础配置 |

---

### 4. `VisDrone2YOLO/` - 数据集目录

VisDrone数据集，已转换为YOLO格式。

**数据统计**:
- 训练集: 6,471 张图片
- 验证集: 548 张图片
- 测试集: 1,610 张图片（可选）
- 类别数: 10 类

**类别列表**:
1. pedestrian（行人）
2. people（人群）
3. bicycle（自行车）
4. car（汽车）
5. van（面包车）
6. truck（卡车）
7. tricycle（三轮车）
8. awning-tricycle（遮阳三轮车）
9. bus（公交车）
10. motor（摩托车）

---

### 5. `results/` - 结果目录

#### 5.1 `results/expected/` - 预期结果文档

每个模型的详细预期结果分析。

| 文档 | 预期mAP50-95 | 说明 |
|------|--------------|------|
| `p2_expected_results.md` | 38-40% | P2模型预期结果 |
| `bifpn_expected_results.md` | 40-43% | BiFPN模型预期结果 |
| `dcn_expected_results.md` | 41-44% | DCN模型预期结果 |
| `carafe_expected_results.md` | 41-44% | CARAFE模型预期结果 |
| `final_expected_results.md` | 42-45% | 最终模型预期结果 |

#### 5.2 实际结果

- `baseline_val/` - Baseline评估结果（图表）
- `baseline_summary.md` - Baseline结果总结
- `p2_val/` - P2评估结果（图表）
- `p2_analysis.md` - P2结果分析

---

### 6. `runs/` - 训练输出目录

Ultralytics自动生成的训练输出。

**典型结构**:
```
runs/visdrone/<实验名>/
├── weights/
│   ├── best.pt          # 最佳权重
│   └── last.pt          # 最后一轮权重
├── results.csv          # 训练结果CSV
├── results.png          # 训练曲线图
├── confusion_matrix.png # 混淆矩阵
└── ...
```

---

### 7. `weights/` - 预训练权重目录

存放预训练权重文件。

| 权重文件 | 说明 |
|----------|------|
| `yolov8s.pt` | YOLOv8s预训练权重 |
| `yolo11n.pt` | YOLO11n预训练权重 |

---

## 🔄 文件移动记录

为了更好的项目组织，以下文件已被移动：

### 文档文件
- `TRAINING_GUIDE.md` → `docs/TRAINING_GUIDE.md`
- `EXPERIMENT_SCHEDULE.md` → `docs/EXPERIMENT_SCHEDULE.md`
- `QUICK_REFERENCE.md` → `docs/QUICK_REFERENCE.md`
- `README_EXPERIMENTS.md` → `docs/README_EXPERIMENTS.md`

### 训练脚本
- `train_*.py` → `scripts/train/train_*.py`

### 评估脚本
- `eval_model.py` → `scripts/eval/eval_model.py`
- `compare_results.py` → `scripts/eval/compare_results.py`
- `check_data.py` → `scripts/eval/check_data.py`

### 推理脚本
- `test_inference.py` → `scripts/inference/test_inference.py`
- `test_ultralytics.py` → `scripts/inference/test_ultralytics.py`

### 权重文件
- `yolov8s.pt` → `weights/yolov8s.pt`
- `yolo11n.pt` → `weights/yolo11n.pt`

---

## 📝 使用建议

### 新手入门流程

1. **阅读主README**: `README.md`
2. **阅读训练指南**: `docs/TRAINING_GUIDE.md`
3. **检查数据**: `python scripts/eval/check_data.py`
4. **开始训练**: `python scripts/train/train_p2.py`
5. **查看预期结果**: `results/expected/p2_expected_results.md`

### 日常使用

- **快速查命令**: `docs/QUICK_REFERENCE.md`
- **查看进度**: `docs/EXPERIMENT_SCHEDULE.md`
- **评估模型**: `python scripts/eval/eval_model.py <model_path>`

---

**最后更新**: 2025-11-02  
**项目版本**: v2.0（重组后）

