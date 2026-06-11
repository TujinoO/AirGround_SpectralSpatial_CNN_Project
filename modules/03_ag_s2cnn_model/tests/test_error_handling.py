"""
错误处理测试模块

测试各种错误处理场景
"""

import pytest
import torch
import numpy as np
import os
import tempfile
from models.ag_s2cnn import AG_S2CNN
from utils.file_utils import (
    validate_file_exists, 
    validate_directory_exists,
    validate_file_extension,
    ensure_directory_exists
)
from train import load_checkpoint


class TestInputValidation:
    """测试输入验证错误处理 (需求 1.3, 1.4, 15.1)"""
    
    def test_invalid_satellite_dimension(self):
        """测试卫星流维度错误"""
        model = AG_S2CNN()
        
        # 错误的空间维度 (15x15 而不是 13x13)
        x_sat = torch.randn(2, 1, 290, 15, 15)
        x_ref = torch.randn(2, 1, 290, 1, 1)
        
        with pytest.raises(ValueError) as exc_info:
            model(x_sat, x_ref)
        
        assert "维度错误" in str(exc_info.value)
        assert "卫星流" in str(exc_info.value)
    
    def test_invalid_ground_dimension(self):
        """测试地面流维度错误"""
        model = AG_S2CNN()
        
        # 错误的地面流维度
        x_sat = torch.randn(2, 1, 290, 13, 13)
        x_ref = torch.randn(2, 1, 290, 3, 3)  # 应该是 1x1
        
        with pytest.raises(ValueError) as exc_info:
            model(x_sat, x_ref)
        
        assert "维度错误" in str(exc_info.value)
        assert "地面流" in str(exc_info.value)
    
    def test_batch_size_mismatch(self):
        """测试批量大小不一致"""
        model = AG_S2CNN()
        
        x_sat = torch.randn(2, 1, 290, 13, 13)
        x_ref = torch.randn(3, 1, 290, 1, 1)  # 不同的批量大小
        
        with pytest.raises(ValueError) as exc_info:
            model(x_sat, x_ref)
        
        assert "批量大小不一致" in str(exc_info.value)
    
    def test_batch_size_out_of_range(self):
        """测试批量大小超出范围"""
        model = AG_S2CNN()
        
        # 批量大小为 0
        x_sat = torch.randn(0, 1, 290, 13, 13)
        x_ref = torch.randn(0, 1, 290, 1, 1)
        
        with pytest.raises(ValueError) as exc_info:
            model(x_sat, x_ref)
        
        assert "批量大小超出范围" in str(exc_info.value)
    
    def test_nan_values_in_satellite(self):
        """测试卫星流包含 NaN 值"""
        model = AG_S2CNN()
        
        x_sat = torch.randn(2, 1, 290, 13, 13)
        x_sat[0, 0, 0, 0, 0] = float('nan')  # 插入 NaN
        x_ref = torch.randn(2, 1, 290, 1, 1)
        
        with pytest.raises(ValueError) as exc_info:
            model(x_sat, x_ref)
        
        assert "NaN" in str(exc_info.value)
        assert "卫星流" in str(exc_info.value)
    
    def test_inf_values_in_ground(self):
        """测试地面流包含 Inf 值"""
        model = AG_S2CNN()
        
        x_sat = torch.randn(2, 1, 290, 13, 13)
        x_ref = torch.randn(2, 1, 290, 1, 1)
        x_ref[0, 0, 0, 0, 0] = float('inf')  # 插入 Inf
        
        with pytest.raises(ValueError) as exc_info:
            model(x_sat, x_ref)
        
        assert "Inf" in str(exc_info.value)
        assert "地面流" in str(exc_info.value)
    
    def test_valid_input(self):
        """测试合法输入"""
        model = AG_S2CNN()
        
        x_sat = torch.randn(2, 1, 290, 13, 13)
        x_ref = torch.randn(2, 1, 290, 1, 1)
        
        # 应该不抛出异常
        output = model(x_sat, x_ref)
        assert output.shape == (2, 4)


class TestFilePathValidation:
    """测试文件路径错误处理 (需求 15.3)"""
    
    def test_file_not_exists(self):
        """测试文件不存在"""
        with pytest.raises(FileNotFoundError) as exc_info:
            validate_file_exists("nonexistent_file.txt", "测试文件")
        
        assert "测试文件不存在" in str(exc_info.value)
        assert "nonexistent_file.txt" in str(exc_info.value)
    
    def test_directory_not_exists(self):
        """测试目录不存在"""
        with pytest.raises(FileNotFoundError) as exc_info:
            validate_directory_exists("nonexistent_dir", "测试目录")
        
        assert "测试目录不存在" in str(exc_info.value)
    
    def test_invalid_file_extension(self):
        """测试文件扩展名错误"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            filepath = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                validate_file_extension(filepath, ['.yaml', '.yml'], "配置文件")
            
            assert "扩展名错误" in str(exc_info.value)
            assert ".txt" in str(exc_info.value)
        finally:
            os.unlink(filepath)
    
    def test_ensure_directory_exists(self):
        """测试确保目录存在"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = os.path.join(tmpdir, 'test_subdir')
            
            # 目录不存在
            assert not os.path.exists(test_dir)
            
            # 创建目录
            ensure_directory_exists(test_dir)
            
            # 目录应该存在
            assert os.path.exists(test_dir)
            assert os.path.isdir(test_dir)
    
    def test_load_checkpoint_file_not_found(self):
        """测试加载不存在的检查点文件"""
        from models.ag_s2cnn import AG_S2CNN
        
        model = AG_S2CNN()
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_checkpoint("nonexistent_checkpoint.pth", model)
        
        assert "检查点文件不存在" in str(exc_info.value)


class TestDimensionAlignment:
    """测试维度对齐错误处理"""
    
    def test_fusion_dimension_mismatch(self):
        """测试融合模块维度不匹配"""
        from models.ag_s2cnn import AGDifferenceFusion
        
        fusion = AGDifferenceFusion()
        
        # 不同的维度
        f_sat = torch.randn(2, 64, 145, 13, 13)
        f_ref = torch.randn(2, 64, 140, 13, 13)  # 不同的光谱维度
        
        with pytest.raises(AssertionError) as exc_info:
            fusion(f_sat, f_ref)
        
        assert "维度不匹配" in str(exc_info.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
