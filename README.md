训练命令（基线 YOLOv8s）  
yolo detect train \
  model=yolov8s.pt \
  data=cfg/visdrone.yaml \
  project=runs/visdrone \
  name=baseline_y8s_1024_adamw \
  cfg=cfg/train_base1024.yaml  

评估与导出  
验证  
yolo detect val \
  model=runs/visdrone/baseline_y8s_1024_adamw/weights/best.pt \
  data=cfg/visdrone.yaml imgsz=1024 \
  project=runs/visdrone name=baseline_eval

导出（部署可选）  
yolo export model=runs/visdrone/baseline_y8s_1024_adamw/weights/best.pt format=onnx half=True

训练与验证命令（P2 版本）
训练（P2）  
yolo detect train \
  model=cfg/models/yolov8s-p2.yaml \
  data=cfg/visdrone.yaml \
  project=runs/visdrone \
  name=y8s_p2_1024_adamw \
  cfg=cfg/train_base1024.yaml

验证  
yolo detect val \
  model=runs/visdrone/y8s_p2_1024_adamw/weights/best.pt \
  data=cfg/visdrone.yaml imgsz=1024 \
  project=runs/visdrone name=y8s_p2_eval