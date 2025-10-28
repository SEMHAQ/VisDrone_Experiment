import os
from pathlib import Path


class PathManager:
    """统一的路径管理器"""

    def __init__(self, project_root=None):
        if project_root is None:
            # 自动检测项目根目录
            self.project_root = self.find_project_root()
        else:
            self.project_root = Path(project_root)

        self.setup_paths()

    def find_project_root(self):
        """自动查找项目根目录"""
        current_file = Path(__file__).resolve()
        # 向上查找包含 configs 和 data 目录的文件夹
        for parent in current_file.parents:
            if (parent / 'configs').exists() and (parent / 'data').exists():
                return parent
        # 如果没找到，使用当前工作目录
        return Path.cwd()

    def setup_paths(self):
        """设置所有路径"""
        # 数据集路径
        self.dataset_root = self.project_root / "VisDrone2019-DET"

        # 配置路径
        self.config_dir = self.project_root / "configs"
        self.dataset_config = self.config_dir / "dataset" / "visdrone.yaml"

        # 模型路径
        self.models_dir = self.project_root / "models"
        self.modules_dir = self.models_dir / "modules"
        self.variants_dir = self.models_dir / "variants"

        # 实验路径
        self.experiments_dir = self.project_root / "experiments"

        # 运行结果路径
        self.runs_dir = self.project_root / "runs"

        # 工具路径
        self.utils_dir = self.project_root / "utils"

        # 脚本路径
        self.scripts_dir = self.project_root / "scripts"

    def validate_paths(self):
        """验证所有必要路径是否存在"""
        required_paths = [
            self.dataset_root,
            self.dataset_config,
            self.dataset_root / "images" / "train",
            self.dataset_root / "images" / "val",
            # 检查转换后的标注路径
            self.dataset_root / "labels" / "train",
            self.dataset_root / "labels" / "val"
        ]

        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(str(path))

        if missing_paths:
            print("❌ 以下路径不存在:")
            for path in missing_paths:
                print(f"   - {path}")

            # 检查是否需要转换标注
            if any("labels" in path for path in missing_paths):
                print("\n⚠ 检测到标注文件未转换")
                print("请运行: python scripts/convert_annotations.py --convert")

            return False
        else:
            print("✅ 所有路径验证通过")
            return True


    def get_experiment_dir(self, exp_name):
        """获取实验目录"""
        return self.runs_dir / exp_name

# 全局路径管理器实例
path_manager = PathManager()