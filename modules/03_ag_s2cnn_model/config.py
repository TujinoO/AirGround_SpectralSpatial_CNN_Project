"""
配置管理模块

提供超参数配置、路径管理和配置验证功能
"""

import os
import yaml
import warnings
from typing import Dict, Any, Optional


class Config:
    """
    配置管理类
    
    功能:
        - 从 YAML 文件加载配置
        - 验证配置完整性
        - 提供默认值
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        'model': {
            'num_bands': 290,
            'spatial_size': 13,
            'num_classes': 1,
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'num_epochs': 100,
            'early_stopping_patience': 20,
            'focal_loss_alpha': 0.75,
            'focal_loss_gamma': 2.0,
            'hnm_ratio': 1.0,
            'hnm_weight': 0.5,
            'separation_weight': 0.5,
            'separation_margin': 0.08,
            'boost': {
                'enabled': False,
                'use_ema': True,
                'use_swa': True,
                'use_mixup': True,
                'use_cutmix': True,
                'use_cosine_lr': True,
                'eval_threshold': 0.5,
                'eval_tta': False,
                'tta_modes': ['none', 'flip_h', 'flip_v', 'flip_hv'],
            },
        },
        'data': {
            'image_path': '',
            'ground_truth_path': '',
            'gsrsl_path': '',
            'output_dir': './outputs',
        },
        'labels': {
            'target_class_name': 'Rich Ore Pegmatite',
            'gt_target_labels': [1],
            'gsrsl_target_labels': [0, 1, 2],
            'background_class_index': 0,
        },
        'augmentation': {
            'enabled': True,
            'noise_std': 0.01,
        },
        'device': {
            'use_cuda': True,
            'gpu_id': 0,
        },
        'visualization': {
            'probability_threshold': 0.6,
            'tta_enabled': False
        }
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化配置
        
        参数:
            config_dict: 配置字典，如果为 None 则使用默认配置
        """
        import copy
        if config_dict is None:
            self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        else:
            self.config = self._merge_with_defaults(config_dict)
        
        self.validate()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        从 YAML 文件加载配置
        
        参数:
            yaml_path: YAML 配置文件路径
        
        返回:
            Config: 配置对象
        
        异常:
            FileNotFoundError: 配置文件不存在
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"配置文件不存在: {os.path.abspath(yaml_path)}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    def _merge_with_defaults(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        将用户配置与默认配置合并
        
        参数:
            config_dict: 用户配置字典
        
        返回:
            合并后的配置字典
        """
        import copy
        merged = copy.deepcopy(self.DEFAULT_CONFIG)
        
        for section, values in config_dict.items():
            if section in merged:
                if isinstance(values, dict):
                    merged[section].update(values)
                else:
                    merged[section] = values
            else:
                merged[section] = values
        
        return merged
    
    def validate(self) -> None:
        """
        验证配置完整性和合法性
        
        异常:
            ValueError: 配置项非法
        """
        # 验证必需的配置节
        required_sections = ['model', 'training', 'data', 'device']
        for section in required_sections:
            if section not in self.config:
                warnings.warn(f"配置节 '{section}' 缺失，使用默认值")
                self.config[section] = self.DEFAULT_CONFIG[section]
        
        # 验证模型配置
        model_config = self.config['model']
        if model_config['num_bands'] <= 0:
            raise ValueError(f"波段数必须为正整数，当前值: {model_config['num_bands']}")
        if model_config['spatial_size'] <= 0 or model_config['spatial_size'] % 2 == 0:
            raise ValueError(f"邻域大小必须为正奇数，当前值: {model_config['spatial_size']}")
        if model_config['num_classes'] <= 0:
            raise ValueError(f"类别数必须为正整数，当前值: {model_config['num_classes']}")
        
        # 验证训练配置
        training_config = self.config['training']
        if training_config['batch_size'] <= 0:
            raise ValueError(f"批量大小必须为正整数，当前值: {training_config['batch_size']}")
        if training_config['learning_rate'] <= 0:
            raise ValueError(f"学习率必须为正数，当前值: {training_config['learning_rate']}")
        if training_config['num_epochs'] <= 0:
            raise ValueError(f"训练轮数必须为正整数，当前值: {training_config['num_epochs']}")
        
        # 验证输出目录
        output_dir = self.config['data']['output_dir']
        if not os.path.exists(output_dir):
            warnings.warn(f"输出目录不存在，将自动创建: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
    
    def save(self, yaml_path: str) -> None:
        """
        保存配置到 YAML 文件
        
        参数:
            yaml_path: 保存路径
        """
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        参数:
            key: 配置键，支持点号分隔的嵌套键（如 'model.num_bands'）
            default: 默认值
        
        返回:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.config[key]
    
    def __repr__(self) -> str:
        return f"Config({self.config})"
