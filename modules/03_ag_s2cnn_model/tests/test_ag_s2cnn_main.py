"""
测试 AG_S2CNN 主模型类

验证主模型的集成、前向传播和输入验证功能
"""

import pytest
import torch
from models.ag_s2cnn import AG_S2CNN


class TestAGS2CNNMain:
    """测试 AG_S2CNN 主模型类"""
    
    def test_model_initialization(self):
        """测试模型初始化"""
        model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
        
        assert model.num_bands == 290
        assert model.spatial_size == 13
        assert model.num_classes == 4
        assert model.reduced_bands == 145  # (290 + 6 - 7) / 2 + 1
        
        # 验证子模块存在
        assert hasattr(model, 'g_encoder')
        assert hasattr(model, 's_backbone')
        assert hasattr(model, 'ag_fusion')
        assert hasattr(model, 'classifier')
    
    def test_forward_pass_basic(self):
        """测试基本前向传播"""
        model = AG_S2CNN()
        batch_size = 4
        
        # 创建输入
        x_sat = torch.randn(batch_size, 1, 290, 13, 13)
        x_ref = torch.randn(batch_size, 1, 290, 1, 1)
        
        # 前向传播
        output = model(x_sat, x_ref)
        
        # 验证输出维度
        assert output.shape == (batch_size, 4)
    
    def test_forward_pass_different_batch_sizes(self):
        """测试不同批量大小的前向传播"""
        model = AG_S2CNN()
        
        for batch_size in [1, 8, 16, 32]:
            x_sat = torch.randn(batch_size, 1, 290, 13, 13)
            x_ref = torch.randn(batch_size, 1, 290, 1, 1)
            
            output = model(x_sat, x_ref)
            
            assert output.shape == (batch_size, 4)
    
    def test_validate_input_correct_dimensions(self):
        """测试正确维度的输入验证"""
        model = AG_S2CNN()
        
        x_sat = torch.randn(4, 1, 290, 13, 13)
        x_ref = torch.randn(4, 1, 290, 1, 1)
        
        # 应该不抛出异常
        model._validate_input(x_sat, x_ref)
    
    def test_validate_input_wrong_sat_dimensions(self):
        """测试错误的卫星流维度"""
        model = AG_S2CNN()
        
        # 错误的空间维度
        x_sat = torch.randn(4, 1, 290, 15, 15)  # 应该是 13x13
        x_ref = torch.randn(4, 1, 290, 1, 1)
        
        with pytest.raises(ValueError, match="卫星流输入期望维度"):
            model._validate_input(x_sat, x_ref)
    
    def test_validate_input_wrong_ref_dimensions(self):
        """测试错误的地面流维度"""
        model = AG_S2CNN()
        
        x_sat = torch.randn(4, 1, 290, 13, 13)
        # 错误的空间维度
        x_ref = torch.randn(4, 1, 290, 3, 3)  # 应该是 1x1
        
        with pytest.raises(ValueError, match="地面流输入期望维度"):
            model._validate_input(x_sat, x_ref)
    
    def test_validate_input_mismatched_batch_size(self):
        """测试批量大小不一致"""
        model = AG_S2CNN()
        
        x_sat = torch.randn(4, 1, 290, 13, 13)
        x_ref = torch.randn(8, 1, 290, 1, 1)  # 不同的批量大小
        
        with pytest.raises(ValueError, match="批量大小不一致"):
            model._validate_input(x_sat, x_ref)
    
    def test_validate_input_batch_size_out_of_range(self):
        """测试批量大小超出范围"""
        model = AG_S2CNN()
        
        # 批量大小为 0
        x_sat = torch.randn(0, 1, 290, 13, 13)
        x_ref = torch.randn(0, 1, 290, 1, 1)
        
        with pytest.raises(ValueError, match="批量大小超出范围"):
            model._validate_input(x_sat, x_ref)
        
        # 批量大小超过 128
        x_sat = torch.randn(129, 1, 290, 13, 13)
        x_ref = torch.randn(129, 1, 290, 1, 1)
        
        with pytest.raises(ValueError, match="批量大小超出范围"):
            model._validate_input(x_sat, x_ref)
    
    def test_validate_input_nan_values(self):
        """测试 NaN 值检测"""
        model = AG_S2CNN()
        
        # 卫星流包含 NaN
        x_sat = torch.randn(4, 1, 290, 13, 13)
        x_sat[0, 0, 0, 0, 0] = float('nan')
        x_ref = torch.randn(4, 1, 290, 1, 1)
        
        with pytest.raises(ValueError, match="卫星流输入包含 NaN 值"):
            model._validate_input(x_sat, x_ref)
        
        # 地面流包含 NaN
        x_sat = torch.randn(4, 1, 290, 13, 13)
        x_ref = torch.randn(4, 1, 290, 1, 1)
        x_ref[0, 0, 0, 0, 0] = float('nan')
        
        with pytest.raises(ValueError, match="地面流输入包含 NaN 值"):
            model._validate_input(x_sat, x_ref)
    
    def test_validate_input_inf_values(self):
        """测试 Inf 值检测"""
        model = AG_S2CNN()
        
        # 卫星流包含 Inf
        x_sat = torch.randn(4, 1, 290, 13, 13)
        x_sat[0, 0, 0, 0, 0] = float('inf')
        x_ref = torch.randn(4, 1, 290, 1, 1)
        
        with pytest.raises(ValueError, match="卫星流输入包含 Inf 值"):
            model._validate_input(x_sat, x_ref)
        
        # 地面流包含 Inf
        x_sat = torch.randn(4, 1, 290, 13, 13)
        x_ref = torch.randn(4, 1, 290, 1, 1)
        x_ref[0, 0, 0, 0, 0] = float('inf')
        
        with pytest.raises(ValueError, match="地面流输入包含 Inf 值"):
            model._validate_input(x_sat, x_ref)
    
    def test_calculate_reduced_bands(self):
        """测试光谱降维计算"""
        model = AG_S2CNN(num_bands=290)
        
        # 验证降维公式: (290 + 2*3 - 7) / 2 + 1 = 145
        assert model.reduced_bands == 145
        
        # 测试其他波段数
        model2 = AG_S2CNN(num_bands=200)
        expected = (200 + 6 - 7) // 2 + 1  # = 100
        assert model2.reduced_bands == expected
    
    def test_end_to_end_forward(self):
        """测试端到端前向传播"""
        model = AG_S2CNN()
        model.eval()  # 设置为评估模式
        
        batch_size = 8
        x_sat = torch.randn(batch_size, 1, 290, 13, 13)
        x_ref = torch.randn(batch_size, 1, 290, 1, 1)
        
        with torch.no_grad():
            output = model(x_sat, x_ref)
        
        # 验证输出
        assert output.shape == (batch_size, 4)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow(self):
        """测试梯度流动"""
        model = AG_S2CNN()
        model.train()
        
        x_sat = torch.randn(4, 1, 290, 13, 13, requires_grad=True)
        x_ref = torch.randn(4, 1, 290, 1, 1, requires_grad=True)
        
        output = model(x_sat, x_ref)
        loss = output.sum()
        loss.backward()
        
        # 验证梯度存在
        assert x_sat.grad is not None
        assert x_ref.grad is not None
        
        # 验证模型参数有梯度
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
