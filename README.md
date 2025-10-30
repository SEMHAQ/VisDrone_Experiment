步骤：  
1.修改configs/dataset/visdrone里的数据集绝对路径  
2.python scripts/verify_environment.py 验证环境  
3.python scripts/convert_annotations.py 转换标签格式(已转换)  
4.python scripts/setup_ablations.py 初始化消融实验框架  
5.python main.py按提示操作即可  (非必须)  
6.python run_ablation_study.py运行四个实验  
7.运行结果在runs文件夹  

运行单个实验步骤：  
python scripts/train_ema_only.py  
python scripts/train_bifpn_only.py  
python scripts/train_full_model.py  
python scripts/verify_experiments.py  

评估四个best.pt:  
python scripts/evaluate_all_models_fixed.py  