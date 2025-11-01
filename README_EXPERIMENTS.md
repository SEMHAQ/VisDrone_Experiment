# VisDrone 数据集实验完整记录

## 实验概述

基于 YOLOv8s 的改进实验，针对 VisDrone 航拍小目标检测数据集进行优化。

### 实验目标
- **基线**: YOLOv8s mAP50-95 = 34.49%
- **目标**: 提升到 **42-45%** (+7.5-10.5 个百分点，+22-30%)
- **重点**: 小目标类别（bicycle, people, pedestrian）提升 **40-70%**

## 实验配置

### 硬件环境
- **GPU**: NVIDIA RTX 3060 12GB
- **Batch Size**: 4-8（根据模型复杂度调整）
- **图像分辨率**: 1024×1024
- **训练轮数**: 200-300 epochs

### 数据集
- **训练集**: 6,471 张图片
- **验证集**: 548 张图片
- **测试集**: 1,610 张图片（可选）
- **类别数**: 10 类（pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor）

## 实验列表

### 1. Baseline (YOLOv8s) ✅
- **模型**: YOLOv8s (标准配置)
- **训练脚本**: `train_baseline.py`
- **结果**: mAP50-95 = 34.49%
- **特点**: 作为对比基线

### 2. P2 Detection Head ✅
- **模型**: YOLOv8s + P2 检测头（stride=4）
- **配置文件**: `cfg/models/yolov8s-p2.yaml`
- **训练脚本**: `train_p2.py`
- **预期结果**: mAP50-95 = 38-40% (+3.5-5.5 个百分点)
- **特点**: 增加 stride=4 检测头，提升小目标检测

### 3. P2 + BiFPN-Lite ✅
- **模型**: YOLOv8s + P2 + BiFPN-Lite
- **配置文件**: `cfg/models/yolov8s-p2-bifpn.yaml`
- **训练脚本**: `train_p2_bifpn.py`
- **预期结果**: mAP50-95 = 40-43% (+5.5-8.5 个百分点)
- **特点**: 双向特征金字塔，改善多尺度特征融合

### 4. P2 + BiFPN + DCNv2 ✅
- **模型**: YOLOv8s + P2 + BiFPN + 少量 DCNv2
- **配置文件**: `cfg/models/yolov8s-p2-bifpn-dcn.yaml`
- **训练脚本**: `train_p2_bifpn_dcn.py`
- **预期结果**: mAP50-95 = 41-44% (+6.5-9.5 个百分点)
- **特点**: 可变形卷积，提升形变和遮挡适应性

### 5. P2 + BiFPN + CARAFE ✅
- **模型**: YOLOv8s + P2 + BiFPN + CARAFE 上采样
- **配置文件**: `cfg/models/yolov8s-p2-bifpn-carafe.yaml`
- **训练脚本**: `train_p2_bifpn_carafe.py`
- **预期结果**: mAP50-95 = 41-44% (+6.5-9.5 个百分点)
- **特点**: 内容感知上采样，提升边界定位精度

### 6. 最终改进模型 ⏳
- **模型**: YOLOv8s + P2 + BiFPN + 小目标友好增强
- **配置文件**: `cfg/models/yolov8s-p2-bifpn-final.yaml`
- **训练脚本**: `train_final_improved.py`
- **预期结果**: mAP50-95 = 42-45% (+7.5-10.5 个百分点)
- **特点**: 
  - P2 检测头
  - BiFPN-Lite 特征融合
  - Copy-Paste 小目标增强 (概率 0.2)
  - 增强调度 (Mosaic 后30%关闭)
  - 充分训练 (300 epochs)

## 训练命令

### 基线模型
```bash
python train_baseline.py
```

### P2 模型
```bash
python train_p2.py
```

### P2 + BiFPN
```bash
python train_p2_bifpn.py
```

### P2 + BiFPN + DCNv2
```bash
python train_p2_bifpn_dcn.py
```

### P2 + BiFPN + CARAFE
```bash
python train_p2_bifpn_carafe.py
```

### 最终改进模型
```bash
python train_final_improved.py
```

## 评估命令

### 对比所有模型
```bash
python compare_results.py
```

### 评估单个模型
```bash
python eval_model.py <模型权重路径>
```

## 预期结果汇总

| 模型 | mAP50-95 | mAP50 | Recall | 相对基线提升 |
|------|----------|-------|--------|--------------|
| Baseline | 34.49% | 53.72% | 50.73% | - |
| P2 | 38-40% | 55-58% | 52-55% | +10-16% |
| P2+BiFPN | 40-43% | 56-59% | 53-56% | +16-25% |
| P2+BiFPN+DCN | 41-44% | 57-60% | 54-57% | +19-28% |
| P2+BiFPN+CARAFE | 41-44% | 57-60% | 54-57% | +19-28% |
| **最终改进** | **42-45%** | **58-61%** | **55-58%** | **+22-30%** |

## 小目标类别详细预期

| 类别 | Baseline | 最终改进模型 | 提升幅度 |
|------|----------|-------------|----------|
| bicycle | 16.84% | **28-35%** | +11-18 个百分点 (+67-108%) |
| people | 24.57% | **35-42%** | +10-17 个百分点 (+42-71%) |
| pedestrian | 32.21% | **41-47%** | +9-15 个百分点 (+27-46%) |
| motor | 31.59% | **40-46%** | +8-14 个百分点 (+27-46%) |

## 论文写作建议

### 实验章节结构

1. **数据集与评估指标**
   - VisDrone 数据集介绍
   - 评估指标定义（mAP50, mAP50-95, Recall）

2. **基线实验**
   - YOLOv8s 在 VisDrone 上的表现
   - 各类别详细分析

3. **消融实验**
   - P2 检测头单独验证
   - BiFPN-Lite 在 P2 基础上验证
   - 小目标友好增强策略验证
   - 每个改进的贡献分析

4. **对比实验**
   - 与其他方法的对比
   - 不同改进组合的对比
   - 计算效率分析

### 图表建议

1. **实验对比表格**: 所有模型的主要指标对比
2. **各类别 AP 对比**: 每种类别的详细提升
3. **PR 曲线**: 小目标类别的 PR 曲线对比
4. **可视化检测结果**: 典型场景的检测对比图
5. **消融实验柱状图**: 每个改进的贡献可视化

## 注意事项

1. **训练时间**: 每个模型训练 200-300 epochs 需要 10-20 小时
2. **显存占用**: 12GB RTX 3060 适配 batch=4
3. **对比公平性**: 所有实验使用相同的训练配置（除改进点外）
4. **结果记录**: 每次实验后运行 `compare_results.py` 记录结果

## 后续工作

- [ ] 完成所有实验训练
- [ ] 运行完整评估
- [ ] 生成论文图表
- [ ] 撰写实验章节
- [ ] 撰写方法章节
- [ ] 撰写结论章节

