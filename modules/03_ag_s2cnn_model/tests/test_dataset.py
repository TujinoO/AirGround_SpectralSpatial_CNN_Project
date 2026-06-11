"""
数据集和数据增强模块测试
"""

import pytest
import torch
import numpy as np
from utils.dataset import HyperspectralDataset, DataAugmentation


class TestHyperspectralDataset:
    """测试 HyperspectralDataset 类"""
    
    def test_dataset_initialization(self):
        """测试数据集初始化"""
        # 创建模拟数据
        h, w, bands = 50, 50, 290
        image_cube = np.random.randn(h, w, bands).astype(np.float32)
        ground_truth = np.zeros((h, w), dtype=np.int32)
        ground_truth[10:20, 10:20] = 1  # 添加一些标签
        ground_truth[30:40, 30:40] = 2
        
        gsrsl = {
            1: np.random.randn(bands).astype(np.float32),
            2: np.random.randn(bands).astype(np.float32)
        }
        
        # 创建数据集
        dataset = HyperspectralDataset(
            image_cube=image_cube,
            ground_truth=ground_truth,
            gsrsl=gsrsl,
            spatial_size=13,
            augmentation=False
        )
        
        # 验证样本数量
        assert len(dataset) > 0
        assert len(dataset) == 200  # 两个 10x10 区域
    
    def test_dataset_getitem(self):
        """测试获取单个样本"""
        # 创建模拟数据
        h, w, bands = 50, 50, 290
        image_cube = np.random.randn(h, w, bands).astype(np.float32)
        ground_truth = np.zeros((h, w), dtype=np.int32)
        ground_truth[20:30, 20:30] = 1
        
        gsrsl = {
            1: np.random.randn(bands).astype(np.float32)
        }
        
        dataset = HyperspectralDataset(
            image_cube=image_cube,
            ground_truth=ground_truth,
            gsrsl=gsrsl,
            spatial_size=13,
            augmentation=False
        )
        
        # 获取样本
        x_sat, x_ref, label = dataset[0]
        
        # 验证维度 (需求 9.3, 9.4, 9.5)
        assert x_sat.shape == (1, bands, 13, 13), f"卫星流维度错误: {x_sat.shape}"
        assert x_ref.shape == (1, bands, 1, 1), f"地面流维度错误: {x_ref.shape}"
        assert label == 1
        
        # 验证数据类型
        assert x_sat.dtype == torch.float32
        assert x_ref.dtype == torch.float32
    
    def test_boundary_padding(self):
        """测试边界填充 (需求 9.6)"""
        # 创建小尺寸影像，确保会触发边界填充
        h, w, bands = 20, 20, 290
        image_cube = np.random.randn(h, w, bands).astype(np.float32)
        ground_truth = np.zeros((h, w), dtype=np.int32)
        ground_truth[0, 0] = 1  # 边界样本
        ground_truth[19, 19] = 2  # 另一个边界样本
        
        gsrsl = {
            1: np.random.randn(bands).astype(np.float32),
            2: np.random.randn(bands).astype(np.float32)
        }
        
        dataset = HyperspectralDataset(
            image_cube=image_cube,
            ground_truth=ground_truth,
            gsrsl=gsrsl,
            spatial_size=13,
            augmentation=False
        )
        
        # 获取边界样本
        x_sat, x_ref, label = dataset[0]
        
        # 验证维度（即使是边界样本，维度也应该正确）
        assert x_sat.shape == (1, bands, 13, 13)
        assert x_ref.shape == (1, bands, 1, 1)
    
    def test_missing_label_in_gsrsl(self):
        """测试标签不在 GSRSL 中的情况"""
        h, w, bands = 30, 30, 290
        image_cube = np.random.randn(h, w, bands).astype(np.float32)
        ground_truth = np.zeros((h, w), dtype=np.int32)
        ground_truth[15, 15] = 3  # 标签 3 不在 GSRSL 中
        
        gsrsl = {
            1: np.random.randn(bands).astype(np.float32),
            2: np.random.randn(bands).astype(np.float32)
        }
        
        dataset = HyperspectralDataset(
            image_cube=image_cube,
            ground_truth=ground_truth,
            gsrsl=gsrsl,
            spatial_size=13,
            augmentation=False
        )
        
        # 获取样本（应该使用零向量作为参考光谱）
        x_sat, x_ref, label = dataset[0]
        
        assert x_ref.shape == (1, bands, 1, 1)
        assert label == 3


class TestDataAugmentation:
    """测试 DataAugmentation 类"""
    
    def test_augmentation_initialization(self):
        """测试数据增强初始化"""
        augmenter = DataAugmentation(noise_std=0.01)
        assert augmenter.noise_std == 0.01
    
    def test_dimension_invariance(self):
        """测试几何变换维度不变性 (需求 7.1, 7.2)"""
        augmenter = DataAugmentation(noise_std=0.01)
        
        # 创建测试数据
        x_sat = torch.randn(1, 290, 13, 13)
        x_ref = torch.randn(1, 290, 1, 1)
        
        # 应用增强
        x_sat_aug, x_ref_aug = augmenter(x_sat, x_ref)
        
        # 验证维度不变
        assert x_sat_aug.shape == x_sat.shape
        assert x_ref_aug.shape == x_ref.shape
    
    def test_ground_flow_invariance(self):
        """测试地面流数据不变性 (需求 7.4)"""
        augmenter = DataAugmentation(noise_std=0.01)
        
        # 创建测试数据
        x_sat = torch.randn(1, 290, 13, 13)
        x_ref = torch.randn(1, 290, 1, 1)
        x_ref_original = x_ref.clone()
        
        # 应用增强
        x_sat_aug, x_ref_aug = augmenter(x_sat, x_ref)
        
        # 验证地面流不变
        assert torch.equal(x_ref_aug, x_ref_original)
    
    def test_satellite_flow_modified(self):
        """测试卫星流被修改"""
        augmenter = DataAugmentation(noise_std=0.01)
        
        # 创建测试数据
        x_sat = torch.randn(1, 290, 13, 13)
        x_ref = torch.randn(1, 290, 1, 1)
        x_sat_original = x_sat.clone()
        
        # 应用增强（多次尝试，因为有随机性）
        modified = False
        for _ in range(10):
            x_sat_test = x_sat_original.clone()
            x_sat_aug, _ = augmenter(x_sat_test, x_ref)
            if not torch.equal(x_sat_aug, x_sat_original):
                modified = True
                break
        
        # 验证卫星流被修改（至少在某次尝试中）
        assert modified, "卫星流应该被数据增强修改"
    
    def test_noise_statistics(self):
        """测试光谱噪声统计特性 (需求 7.3)"""
        augmenter = DataAugmentation(noise_std=0.01)
        
        # 创建测试数据（全零，便于观察噪声）
        x_sat = torch.zeros(1, 290, 13, 13)
        
        # 仅应用噪声（不应用几何变换）
        x_noisy = augmenter._add_spectral_noise(x_sat)
        
        # 验证噪声的统计特性
        noise = x_noisy - x_sat
        mean = noise.mean().item()
        std = noise.std().item()
        
        # 均值应接近 0（允许一定误差）
        assert abs(mean) < 0.01, f"噪声均值应接近 0，实际为 {mean}"
        
        # 标准差应接近 0.01（允许一定误差）
        assert abs(std - 0.01) < 0.005, f"噪声标准差应接近 0.01，实际为 {std}"


class TestDatasetWithAugmentation:
    """测试数据集与数据增强的集成"""
    
    def test_dataset_with_augmentation(self):
        """测试启用数据增强的数据集"""
        h, w, bands = 50, 50, 290
        image_cube = np.random.randn(h, w, bands).astype(np.float32)
        ground_truth = np.zeros((h, w), dtype=np.int32)
        ground_truth[20:30, 20:30] = 1
        
        gsrsl = {
            1: np.random.randn(bands).astype(np.float32)
        }
        
        # 创建启用增强的数据集
        dataset = HyperspectralDataset(
            image_cube=image_cube,
            ground_truth=ground_truth,
            gsrsl=gsrsl,
            spatial_size=13,
            augmentation=True
        )
        
        # 获取样本
        x_sat, x_ref, label = dataset[0]
        
        # 验证维度
        assert x_sat.shape == (1, bands, 13, 13)
        assert x_ref.shape == (1, bands, 1, 1)
        assert label == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
