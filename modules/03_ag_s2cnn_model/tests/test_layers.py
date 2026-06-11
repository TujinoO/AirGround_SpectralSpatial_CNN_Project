"""
测试基础层和工具模块

测试 ResNetBlock2D 和 Inception3D 的基本功能
"""

import pytest
import torch
from models.layers import ResNetBlock2D, Inception3D


class TestResNetBlock2D:
    """测试 ResNetBlock2D 残差卷积块"""
    
    def test_output_shape(self):
        """测试输出维度是否与输入维度一致"""
        batch_size = 4
        channels = 64
        height = 13
        width = 13
        
        block = ResNetBlock2D(channels)
        x = torch.randn(batch_size, channels, height, width)
        
        output = block(x)
        
        assert output.shape == (batch_size, channels, height, width)
    
    def test_residual_connection(self):
        """测试残差连接是否正确"""
        channels = 32
        block = ResNetBlock2D(channels)
        
        # 使用零初始化权重来测试残差连接
        with torch.no_grad():
            # 将第二层卷积的权重设为零，这样输出应该接近输入
            block.conv2.weight.zero_()
            block.conv2.bias.zero_()
        
        x = torch.randn(2, channels, 5, 5)
        output = block(x)
        
        # 输出应该不等于输入（因为有第一层卷积和激活）
        assert not torch.allclose(output, x)
        
        # 但输出维度应该相同
        assert output.shape == x.shape
    
    def test_different_batch_sizes(self):
        """测试不同批量大小的处理"""
        channels = 128
        block = ResNetBlock2D(channels)
        
        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, channels, 13, 13)
            output = block(x)
            assert output.shape == (batch_size, channels, 13, 13)


class TestInception3D:
    """测试 Inception3D 多尺度模块"""
    
    def test_output_channels(self):
        """测试输出通道数为 48（24+24）"""
        batch_size = 4
        in_channels = 16
        depth = 145
        height = 13
        width = 13
        
        inception = Inception3D(in_channels)
        x = torch.randn(batch_size, in_channels, depth, height, width)
        
        output = inception(x)
        
        # 输出通道数应该是 48（两个分支各 24）
        assert output.shape == (batch_size, 48, depth, height, width)
    
    def test_spatial_dimensions_preserved(self):
        """测试空间维度保持不变"""
        in_channels = 16
        inception = Inception3D(in_channels)
        
        x = torch.randn(2, in_channels, 100, 13, 13)
        output = inception(x)
        
        # 空间维度应该保持不变
        assert output.shape[2:] == (100, 13, 13)
    
    def test_different_batch_sizes(self):
        """测试不同批量大小的处理"""
        in_channels = 16
        inception = Inception3D(in_channels)
        
        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, in_channels, 145, 13, 13)
            output = inception(x)
            assert output.shape == (batch_size, 48, 145, 13, 13)
    
    def test_branch_concatenation(self):
        """测试两个分支的输出是否正确拼接"""
        in_channels = 16
        inception = Inception3D(in_channels)
        
        x = torch.randn(2, in_channels, 50, 13, 13)
        output = inception(x)
        
        # 验证输出是两个分支的拼接
        # 每个分支输出 24 个通道，总共 48 个通道
        assert output.shape[1] == 48
        
        # 验证输出不是全零（说明两个分支都有贡献）
        assert not torch.allclose(output, torch.zeros_like(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
