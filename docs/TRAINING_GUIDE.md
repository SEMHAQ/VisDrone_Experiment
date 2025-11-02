# 🚀 VisDrone 实验训练指南

## 📊 当前状态
- ✅ **Baseline**: 已完成，mAP50-95 = 34.49%
- ⚠️ **P2**: 需重训练（配置已修复）
- ⏳ **后续实验**: 待执行

---

## 🎯 推荐实验路线

### **方案A: 核心实验（推荐，3个实验）**
适合时间有限，只做核心改进验证

1. ✅ Baseline (已完成)
2. ⏳ P2 模型
3. ⏳ P2 + BiFPN
4. ⏳ 最终改进模型

**总时间**: ~66小时（约3天）

---

### **方案B: 完整实验（5个实验）**
适合做完整消融实验，论文更充分

1. ✅ Baseline (已完成)
2. ⏳ P2 模型
3. ⏳ P2 + BiFPN
4. ⏳ P2 + BiFPN + DCN（或CARAFE）
5. ⏳ 最终改进模型

**总时间**: ~90小时（约4天）

---

## 📝 实验详细说明

### 实验1: Baseline ✅
```bash
# 已完成，无需重新训练
```
- **结果**: mAP50-95 = 34.49%
- **参考**: `results/baseline_summary.md`

---

### 实验2: P2模型 ⚠️ **需重训练**

#### 训练命令
```bash
python train_p2.py
```

#### 配置信息
- **Epochs**: 300（已修复，原来是30）
- **Batch Size**: 4
- **学习率**: 0.002
- **训练时间**: ~20小时

#### 预期结果
- **mAP50-95**: 38-40%（+3.5-5.5个百分点）
- **bicycle**: 22-28%（+5-11个百分点）
- **people**: 29-35%（+4-10个百分点）

#### 成功标准
- ✅ mAP50-95 > 38%
- ✅ bicycle AP > 20%
- ✅ people AP > 28%

#### 参考文档
- `results/expected/p2_expected_results.md`

---

### 实验3: P2 + BiFPN

#### 训练命令
```bash
python train_p2_bifpn.py
```

#### 配置信息
- **Epochs**: 200
- **Batch Size**: 4
- **学习率**: 0.001
- **训练时间**: ~22小时

#### 预期结果
- **mAP50-95**: 40-43%（+5.5-8.5个百分点）
- **bicycle**: 25-32%（+8-15个百分点）
- **people**: 32-38%（+7-13个百分点）

#### 成功标准
- ✅ mAP50-95 > 40%
- ✅ 相比P2提升 > 1.5个百分点
- ✅ bicycle AP > 25%

#### 参考文档
- `results/expected/bifpn_expected_results.md`

---

### 实验4A: P2 + BiFPN + DCN（可选）

#### 训练命令
```bash
python train_p2_bifpn_dcn.py
```

#### 配置信息
- **Epochs**: 200
- **Batch Size**: 4（如果OOM降到3）
- **学习率**: 0.001
- **训练时间**: ~24小时

#### 预期结果
- **mAP50-95**: 41-44%（+6.5-9.5个百分点）
- **遮挡场景**: 提升明显

#### 注意事项
- ⚠️ 需要DCNv2依赖，如果安装困难可跳过
- ⚠️ 显存占用较大，可能需要降低batch

#### 参考文档
- `results/expected/dcn_expected_results.md`

---

### 实验4B: P2 + BiFPN + CARAFE（可选）

#### 训练命令
```bash
python train_p2_bifpn_carafe.py
```

#### 配置信息
- **Epochs**: 200
- **Batch Size**: 4
- **学习率**: 0.002
- **训练时间**: ~22小时

#### 预期结果
- **mAP50-95**: 41-44%（+6.5-9.5个百分点）
- **边界定位**: 提升明显

#### 参考文档
- `results/expected/carafe_expected_results.md`

---

### 实验5: 最终改进模型

#### 训练命令
```bash
python train_final_improved.py
```

#### 配置信息
- **Epochs**: 300
- **Batch Size**: 4
- **学习率**: 0.001
- **训练时间**: ~24小时
- **关键改进**: 
  - P2 检测头
  - BiFPN-Lite
  - Copy-Paste (0.2)
  - Mosaic调度（后30%关闭）

#### 预期结果
- **mAP50-95**: 42-45%（+7.5-10.5个百分点，+22-30%）
- **bicycle**: 28-35%（+11-18个百分点，+67-108%）
- **people**: 35-42%（+10-17个百分点，+42-71%）

#### 成功标准
- ✅ mAP50-95 > 42%
- ✅ bicycle AP > 28%
- ✅ people AP > 35%
- ✅ 相对baseline提升 > 22%

#### 参考文档
- `results/expected/final_expected_results.md`

---

## 🔧 训练监控

### 1. GPU监控
```bash
# 新开终端，实时监控GPU
nvidia-smi -l 1
```

### 2. 查看训练日志
```bash
# 查看最新训练的结果
cat runs/visdrone/<实验名>/results.csv
```

### 3. TensorBoard（可选）
```bash
tensorboard --logdir runs/visdrone
```

---

## 📊 评估命令

### 单模型评估
```bash
python eval_model.py runs/visdrone/<实验名>/weights/best.pt
```

### 对比所有模型
```bash
python compare_results.py
```

---

## ⚠️ 常见问题

### Q1: 显存不足（OOM）
**解决方案**:
```python
# 方案1: 降低batch size
batch = 3  # 或 2

# 方案2: 降低图像尺寸
imgsz = 960  # 从1024降到960

# 方案3: 关闭其他程序
# 检查是否有其他程序占用显存
```

### Q2: 训练中断
**解决方案**:
```bash
# 从checkpoint恢复
python train_p2.py --resume runs/visdrone/<实验名>/weights/last.pt
```

### Q3: 结果不理想
**检查清单**:
- [ ] 训练是否收敛（查看loss曲线）
- [ ] Epochs是否足够（至少200）
- [ ] 学习率是否合适
- [ ] 数据是否正确加载

---

## 📈 预期结果汇总

| 实验 | mAP50-95 | bicycle | people | 训练时间 | 状态 |
|------|----------|---------|--------|----------|------|
| Baseline | 34.49% | 16.84% | 24.57% | - | ✅ |
| P2 | 38-40% | 22-28% | 29-35% | 20h | ⚠️ 重做 |
| P2+BiFPN | 40-43% | 25-32% | 32-38% | 22h | ⏳ |
| P2+BiFPN+DCN | 41-44% | 26-33% | 33-39% | 24h | ⏳ 可选 |
| P2+BiFPN+CARAFE | 41-44% | 26-33% | 33-39% | 22h | ⏳ 可选 |
| **最终模型** | **42-45%** | **28-35%** | **35-42%** | 24h | ⏳ |

---

## 🎯 下一步行动

### 立即执行（今天）
```bash
# 1. 重新训练P2模型
python train_p2.py

# 2. 监控训练（新开终端）
nvidia-smi -l 1
```

### 明天
- 检查P2训练结果
- 如果达到预期（mAP > 38%），开始BiFPN训练

### 后天
- 检查BiFPN训练结果
- 决定是否做DCN/CARAFE实验

### 3-4天后
- 开始最终改进模型训练
- 准备论文图表

---

## 📁 重要文件位置

### 训练脚本
- `train_baseline.py` - Baseline（已完成）
- `train_p2.py` - P2模型（需重训练）
- `train_p2_bifpn.py` - P2+BiFPN
- `train_p2_bifpn_dcn.py` - P2+BiFPN+DCN（可选）
- `train_p2_bifpn_carafe.py` - P2+BiFPN+CARAFE（可选）
- `train_final_improved.py` - 最终模型

### 配置文件
- `cfg/visdrone.yaml` - 数据配置
- `cfg/models/yolov8s-p2.yaml` - P2模型
- `cfg/models/yolov8s-p2-bifpn.yaml` - BiFPN模型
- `cfg/models/yolov8s-p2-bifpn-dcn.yaml` - DCN模型
- `cfg/models/yolov8s-p2-bifpn-carafe.yaml` - CARAFE模型
- `cfg/models/yolov8s-p2-bifpn-final.yaml` - 最终模型

### 预期结果
- `results/expected/p2_expected_results.md`
- `results/expected/bifpn_expected_results.md`
- `results/expected/dcn_expected_results.md`
- `results/expected/carafe_expected_results.md`
- `results/expected/final_expected_results.md`

### 实际结果
- `results/baseline_summary.md` - Baseline结果
- `runs/visdrone/` - 所有训练输出

---

## ✅ 检查清单

### 训练前
- [ ] 数据集完整（train: 6471张，val: 548张）
- [ ] GPU可用（nvidia-smi检查）
- [ ] 配置文件存在
- [ ] 训练脚本可执行

### 训练中
- [ ] GPU利用率正常（80-100%）
- [ ] Loss正常下降
- [ ] 显存未溢出

### 训练后
- [ ] 评估结果
- [ ] 对比预期结果
- [ ] 保存最佳权重
- [ ] 记录实验结果

---

**创建时间**: 2025-11-02  
**最后更新**: 2025-11-02  
**当前任务**: P2模型重训练

