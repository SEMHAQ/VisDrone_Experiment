步骤：  
1.修改configs/dataset/visdrone里的数据集绝对路径  
2.python scripts/verify_environment.py 验证环境  
3.python scripts/convert_annotations.py 转换标签格式  
4.python scripts/setup_ablations.py 初始化消融实验框架  
5.python main.py按提示操作即可  
6.python run_ablation_study.py运行四个实验  
7.运行结果在runs文件夹

