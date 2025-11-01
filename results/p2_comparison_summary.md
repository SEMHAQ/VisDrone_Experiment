# P2 模型性能对比

## 评估结果

### Baseline (YOLOv8s)

- mAP50: 0.5372
- mAP50-95: 0.3449
- Precision: 0.6547
- Recall: 0.5073
- F1: 0.0000

#### 各类别 mAP50-95

- pedestrian: 0.3221
- people: 0.2457
- bicycle: 0.1684
- car: 0.6257
- van: 0.4304
- truck: 0.3698
- tricycle: 0.2757
- awning-tricycle: 0.1680
- bus: 0.5279
- motor: 0.3159

### P2 Model (YOLOv8s-P2)

- mAP50: 0.4804
- mAP50-95: 0.2997
- Precision: 0.6241
- Recall: 0.4969
- F1: 0.0000

#### 各类别 mAP50-95

- pedestrian: 0.2991
- people: 0.2154
- bicycle: 0.1135
- car: 0.5979
- van: 0.3918
- truck: 0.2873
- tricycle: 0.2264
- awning-tricycle: 0.1434
- bus: 0.4532
- motor: 0.2692

## 对比分析

总体提升: -0.0452 mAP50-95
