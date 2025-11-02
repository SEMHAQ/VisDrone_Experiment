# 🚀 VisDrone 实验快速参考

## 📊 当前状态
- ✅ Baseline: 34.49% mAP50-95
- ⚠️ P2: 需重训练（配置已修复）
- 🎯 目标: 42-45% mAP50-95

---

## 🔥 立即执行

### 1. 重新训练P2模型（已修复配置）
```bash
python train_p2.py
```
**预计时间**: 20小时  
**预期结果**: mAP50-95 = 38-40%

### 2. 监控训练
```bash
# 查看GPU使用
nvidia-smi -l 1

# TensorBoard（可选）
tensorboard --logdir runs/visdrone
```

### 3. 训练完成后评估
```bash
python eval_model.py runs/visdrone/y8s_p2_1024_adamw_300ep/weights/best.pt
```

---

## 📋 实验检查清单

### 阶段1: 基础 ✅
- [x] Baseline训练 (34.49%)

### 阶段2: 架构改进
- [ ] P2重训练 (目标: 38-40%)
- [ ] P2+BiFPN (目标: 40-42%)
- [ ] P2+BiFPN+DCN (可选, 目标: 41-43%)
- [ ] P2+BiFPN+CARAFE (可选, 目标: 41-43%)

### 阶段3: 细节优化
- [ ] Head注意力 (+0.5-1.5%)
- [ ] 损失优化 (+0.5-1.0%)
- [ ] 小目标增强 (+1.0-2.0%)

### 阶段4: 推理优化
- [ ] Tiling推理 (Recall +3-5%)
- [ ] Soft-NMS + TTA (+0.5-1.5%)

### 阶段5: 评估与论文
- [ ] 实验评估与可视化
- [ ] 论文图表制作

---

## ⚙️ 关键配置速查

### Baseline
```python
batch=6, epochs=300, lr=0.002, imgsz=1024
```

### P2（已修复）
```python
batch=4, epochs=300, lr=0.002, warmup=15, imgsz=1024
```

### P2+BiFPN
```python
batch=4, epochs=300, lr=0.002, warmup=15, imgsz=1024
```

### P2+BiFPN+DCN
```python
batch=3-4, epochs=300, lr=0.001, warmup=15, imgsz=1024
```

---

## 🎯 目标指标

| 类别 | Baseline | 目标 | 提升 |
|------|----------|------|------|
| **整体** | 34.49% | 42-45% | +22-30% |
| bicycle | 16.84% | 28-35% | +67-108% |
| people | 24.57% | 35-42% | +42-71% |
| pedestrian | 32.21% | 41-47% | +27-46% |

---

## 💾 显存优化（12GB RTX 3060）

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

## 📁 重要文件

### 训练脚本
- `train_baseline.py` - Baseline
- `train_p2.py` - P2（已修复）
- `train_p2_bifpn.py` - P2+BiFPN
- `train_p2_bifpn_dcn.py` - P2+BiFPN+DCN
- `train_p2_bifpn_carafe.py` - P2+BiFPN+CARAFE
- `train_final_improved.py` - 最终模型

### 配置文件
- `cfg/visdrone.yaml` - 数据配置
- `cfg/models/yolov8s-p2.yaml` - P2模型
- `cfg/models/yolov8s-p2-bifpn.yaml` - BiFPN模型
- `cfg/models/yolov8s-p2-bifpn-dcn.yaml` - DCN模型
- `cfg/models/yolov8s-p2-bifpn-final.yaml` - 最终模型

### 评估脚本
- `eval_model.py` - 单模型评估
- `compare_results.py` - 多模型对比

### 结果目录
- `results/` - 评估结果和分析
- `runs/visdrone/` - 训练输出

---

## 🔧 常用命令

### 训练
```bash
# P2模型（当前任务）
python train_p2.py

# BiFPN模型（下一步）
python train_p2_bifpn.py

# 从checkpoint恢复
python train_p2.py --resume runs/visdrone/xxx/weights/last.pt
```

### 评估
```bash
# 单模型评估
python eval_model.py <model_path>

# 对比所有模型
python compare_results.py
```

### 监控
```bash
# GPU监控
nvidia-smi -l 1

# 查看训练日志
cat runs/visdrone/xxx/results.csv

# TensorBoard
tensorboard --logdir runs/visdrone
```

---

## ⏱️ 时间估算

| 任务 | 时间 | 累计 |
|------|------|------|
| P2重训练 | 20h | 20h |
| P2+BiFPN | 22h | 42h |
| P2+BiFPN+优化 | 24h | 66h |
| 推理优化 | 5h | 71h |
| 评估分析 | 10h | 81h |

**总计**: 约3-4周（包含调试时间）

---

## 📞 问题排查

### 训练不收敛
- 检查学习率是否过大
- 增加warmup epochs
- 检查数据是否正确加载

### 显存不足
- 降低batch size
- 启用梯度累积
- 关闭不必要的程序

### 性能下降
- 检查训练epochs是否足够
- 对比训练配置是否一致
- 查看验证集loss曲线

---

## 📈 成功标准

### P2模型（当前）
- [x] mAP50-95 > 38%
- [x] bicycle AP > 20%
- [x] people AP > 28%
- [x] 训练收敛（loss稳定）

### 最终模型
- [x] mAP50-95 > 42%
- [x] bicycle AP > 28%
- [x] people AP > 35%
- [x] 相对baseline提升 > 22%

---

## 🎓 论文写作提示

### 实验章节结构
1. 数据集与评估指标
2. 基线实验
3. 消融实验（P2, BiFPN, 增强等）
4. 对比实验
5. 可视化分析

### 必需图表
- [ ] 实验对比表格
- [ ] 消融实验表格
- [ ] 各类别AP对比
- [ ] PR曲线
- [ ] 可视化检测结果
- [ ] 训练曲线

---

**最后更新**: 2025-11-02  
**下次检查**: P2训练完成后

