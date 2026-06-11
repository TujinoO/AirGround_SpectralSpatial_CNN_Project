"""
测试特征提取模块

测试 G-Encoder 和 S-Backbone 的功能和维度对齐
"""

import pytest
import torch
from models.ag_s2cnn import GEncoder, SBackbone


class TestGEncoder:
    """测试 G-Encoder 地面光谱编码器"""
    
    def test_output_shape(self):
        """测试输出维度为 (N, 64, D_new, 13, 13)"""
        batch_size = 4
        num_bands = 290
        spatial_size = 13
        
        encoder = GEncoder(num_bands=num_bands, spatial_size=spatial_size)
        x_ref = torch.randn(batch_size, 1, num_bands, 1, 1)
        
        output = encoder(x_ref)
        
        # 验证输出维度
        assert output.shape[0] == batch_size  # 批量维度
        assert output.shape[1] == 64  # 通道数
        assert output.shape[3] == spatial_size  # 空间高度
        assert output.shape[4] == spatial_size  # 空间宽度
        
        # 验证光谱降维（290 → 145）
        expected_depth = (num_bands + 2 * 3 - 7) // 2 + 1  # 考虑 padding=3, kernel=7, stride=2
        assert output.shape[2] == expected_depth
    
    def test_spatial_broadcasting(self):
        """测试 Spatial Broadcasting 机制"""
        batch_size = 2
        encoder = GEncoder(num_bands=290, spatial_size=13)
        x_ref = torch.randn(batch_size, 1, 290, 1, 1)
        
        output = encoder(x_ref)
        
        # 验证空间维度从 1×1 扩展到 13×13
        assert output.shape[3] == 13
        assert output.shape[4] == 13
    
    def test_different_batch_sizes(self):
        """测试不同批量大小的处理"""
        encoder = GEncoder(num_bands=290, spatial_size=13)
        
        for batch_size in [1, 8, 16, 32, 64, 128]:
            x_ref = torch.randn(batch_size, 1, 290, 1, 1)
            output = encoder(x_ref)
            
            assert output.shape[0] == batch_size
            assert output.shape[1] == 64
            assert output.shape[3] == 13
            assert output.shape[4] == 13
    
    def test_spectral_reduction(self):
        """测试光谱降维是否正确（290 → 145）"""
        encoder = GEncoder(num_bands=290, spatial_size=13)
        x_ref = torch.randn(1, 1, 290, 1, 1)
        
        output = encoder(x_ref)
        
        # 验证光谱维度降维
        assert output.shape[2] == 145


class TestSBackbone:
    """测试 S-Backbone 卫星特征提取主干"""
    
    def test_output_shape(self):
        """测试输出维度为 (N, 64, D_new, 13, 13)"""
        batch_size = 4
        num_bands = 290
        spatial_size = 13
        
        backbone = SBackbone(num_bands=num_bands, spatial_size=spatial_size)
        x_sat = torch.randn(batch_size, 1, num_bands, spatial_size, spatial_size)
        
        output = backbone(x_sat)
        
        # 验证输出维度
        assert output.shape[0] == batch_size  # 批量维度
        assert output.shape[1] == 64  # 通道数
        assert output.shape[3] == spatial_size  # 空间高度
        assert output.shape[4] == spatial_size  # 空间宽度
        
        # 验证光谱降维（290 → 145）
        expected_depth = (num_bands + 2 * 3 - 7) // 2 + 1  # 考虑 padding=3, kernel=7, stride=2
        assert output.shape[2] == expected_depth
    
    def test_spatial_dimensions_preserved(self):
        """测试空间维度保持不变（13×13）"""
        backbone = SBackbone(num_bands=290, spatial_size=13)
        x_sat = torch.randn(2, 1, 290, 13, 13)
        
        output = backbone(x_sat)
        
        # 验证空间维度不变
        assert output.shape[3] == 13
        assert output.shape[4] == 13
    
    def test_different_batch_sizes(self):
        """测试不同批量大小的处理"""
        backbone = SBackbone(num_bands=290, spatial_size=13)
        
        for batch_size in [1, 8, 16, 32]:
            x_sat = torch.randn(batch_size, 1, 290, 13, 13)
            output = backbone(x_sat)
            
            assert output.shape[0] == batch_size
            assert output.shape[1] == 64
            assert output.shape[3] == 13
            assert output.shape[4] == 13
    
    def test_spectral_reduction(self):
        """测试光谱降维是否正确"""
        backbone = SBackbone(num_bands=290, spatial_size=13)
        x_sat = torch.randn(1, 1, 290, 13, 13)
        
        output = backbone(x_sat)
        
        # 验证光谱维度降维
        assert output.shape[2] == 145
    
    def test_inception_integration(self):
        """测试 Inception-3D 模块集成"""
        backbone = SBackbone(num_bands=290, spatial_size=13)
        x_sat = torch.randn(2, 1, 290, 13, 13)
        
        output = backbone(x_sat)
        
        # 验证输出不是全零（说明 Inception 模块有贡献）
        assert not torch.allclose(output, torch.zeros_like(output))


class TestDimensionAlignment:
    """测试 G-Encoder 和 S-Backbone 的维度对齐"""
    
    def test_output_dimensions_match(self):
        """
        验证 G-Encoder 和 S-Backbone 输出维度一致
        需求: 2.6, 3.7, 12.1, 12.2, 12.3
        """
        batch_size = 8
        num_bands = 290
        spatial_size = 13
        
        # 初始化两个模块
        g_encoder = GEncoder(num_bands=num_bands, spatial_size=spatial_size)
        s_backbone = SBackbone(num_bands=num_bands, spatial_size=spatial_size)
        
        # 准备输入
        x_ref = torch.randn(batch_size, 1, num_bands, 1, 1)
        x_sat = torch.randn(batch_size, 1, num_bands, spatial_size, spatial_size)
        
        # 前向传播
        f_ref = g_encoder(x_ref)
        f_sat = s_backbone(x_sat)
        
        # 验证维度完全一致
        assert f_ref.shape == f_sat.shape, \
            f"维度不匹配: G-Encoder 输出 {f_ref.shape} vs S-Backbone 输出 {f_sat.shape}"
        
        # 验证具体维度
        assert f_ref.shape == (batch_size, 64, 145, spatial_size, spatial_size)
        assert f_sat.shape == (batch_size, 64, 145, spatial_size, spatial_size)
    
    def test_spectral_dimension_consistency(self):
        """验证光谱维度 D_new 在两个模块中一致"""
        g_encoder = GEncoder(num_bands=290, spatial_size=13)
        s_backbone = SBackbone(num_bands=290, spatial_size=13)
        
        x_ref = torch.randn(4, 1, 290, 1, 1)
        x_sat = torch.randn(4, 1, 290, 13, 13)
        
        f_ref = g_encoder(x_ref)
        f_sat = s_backbone(x_sat)
        
        # 验证光谱维度（索引 2）一致
        assert f_ref.shape[2] == f_sat.shape[2], \
            f"光谱维度不一致: G-Encoder {f_ref.shape[2]} vs S-Backbone {f_sat.shape[2]}"
    
    def test_channel_dimension_consistency(self):
        """验证通道维度在两个模块中一致（都是 64）"""
        g_encoder = GEncoder(num_bands=290, spatial_size=13)
        s_backbone = SBackbone(num_bands=290, spatial_size=13)
        
        x_ref = torch.randn(2, 1, 290, 1, 1)
        x_sat = torch.randn(2, 1, 290, 13, 13)
        
        f_ref = g_encoder(x_ref)
        f_sat = s_backbone(x_sat)
        
        # 验证通道维度（索引 1）一致
        assert f_ref.shape[1] == 64
        assert f_sat.shape[1] == 64
        assert f_ref.shape[1] == f_sat.shape[1]
    
    def test_spatial_dimension_consistency(self):
        """验证空间维度在两个模块中一致（都是 13×13）"""
        spatial_size = 13
        g_encoder = GEncoder(num_bands=290, spatial_size=spatial_size)
        s_backbone = SBackbone(num_bands=290, spatial_size=spatial_size)
        
        x_ref = torch.randn(2, 1, 290, 1, 1)
        x_sat = torch.randn(2, 1, 290, spatial_size, spatial_size)
        
        f_ref = g_encoder(x_ref)
        f_sat = s_backbone(x_sat)
        
        # 验证空间维度（索引 3 和 4）一致
        assert f_ref.shape[3] == spatial_size
        assert f_ref.shape[4] == spatial_size
        assert f_sat.shape[3] == spatial_size
        assert f_sat.shape[4] == spatial_size
    
    def test_multiple_batch_sizes_alignment(self):
        """测试不同批量大小下的维度对齐"""
        g_encoder = GEncoder(num_bands=290, spatial_size=13)
        s_backbone = SBackbone(num_bands=290, spatial_size=13)
        
        for batch_size in [1, 4, 16, 32, 64]:
            x_ref = torch.randn(batch_size, 1, 290, 1, 1)
            x_sat = torch.randn(batch_size, 1, 290, 13, 13)
            
            f_ref = g_encoder(x_ref)
            f_sat = s_backbone(x_sat)
            
            # 验证维度完全一致
            assert f_ref.shape == f_sat.shape, \
                f"批量大小 {batch_size} 时维度不匹配"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
