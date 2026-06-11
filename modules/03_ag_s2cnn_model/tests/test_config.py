"""
配置管理模块测试
"""

import os
import pytest
import tempfile
import yaml
from config import Config


def test_default_config():
    """测试默认配置"""
    config = Config()
    
    # 验证默认值
    assert config.get('model.num_bands') == 290
    assert config.get('model.spatial_size') == 13
    assert config.get('model.num_classes') == 4
    assert config.get('training.batch_size') == 32
    assert config.get('training.learning_rate') == 1e-3


def test_config_from_yaml():
    """测试从 YAML 文件加载配置"""
    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'model': {'num_bands': 200},
            'training': {'batch_size': 64}
        }, f)
        temp_path = f.name
    
    try:
        config = Config.from_yaml(temp_path)
        
        # 验证自定义值
        assert config.get('model.num_bands') == 200
        assert config.get('training.batch_size') == 64
        
        # 验证默认值仍然存在
        assert config.get('model.spatial_size') == 13
    finally:
        os.unlink(temp_path)


def test_config_validation():
    """测试配置验证"""
    # 测试非法波段数
    with pytest.raises(ValueError, match="波段数必须为正整数"):
        Config({'model': {'num_bands': -1, 'spatial_size': 13, 'num_classes': 4}})
    
    # 测试非法邻域大小（偶数）
    with pytest.raises(ValueError, match="邻域大小必须为正奇数"):
        Config({'model': {'num_bands': 290, 'spatial_size': 12, 'num_classes': 4}})
    
    # 测试非法批量大小
    with pytest.raises(ValueError, match="批量大小必须为正整数"):
        Config({'training': {'batch_size': 0}})


def test_config_file_not_found():
    """测试配置文件不存在"""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml('nonexistent_config.yaml')


def test_config_save():
    """测试保存配置"""
    config = Config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        config.save(temp_path)
        
        # 验证文件存在
        assert os.path.exists(temp_path)
        
        # 验证可以重新加载
        loaded_config = Config.from_yaml(temp_path)
        assert loaded_config.get('model.num_bands') == 290
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_config_get_nested():
    """测试嵌套配置访问"""
    config = Config()
    
    # 测试点号分隔的键
    assert config.get('model.num_bands') == 290
    assert config.get('training.learning_rate') == 1e-3
    
    # 测试不存在的键
    assert config.get('nonexistent.key', 'default') == 'default'


def test_config_dict_access():
    """测试字典式访问"""
    config = Config()
    
    assert config['model']['num_bands'] == 290
    assert config['training']['batch_size'] == 32
