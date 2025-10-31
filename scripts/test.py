
import os

# 依次运行三个实验
os.system("python train_ema_only.py")
os.system("python train_bifpn_only.py")
os.system("python train_ema_bifpn.py")