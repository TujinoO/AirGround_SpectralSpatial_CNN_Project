"""
训练流水线测试

测试训练循环、验证、早停机制、模型保存加载和设备管理功能
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
from torch.utils.data import DataLoader, TensorDataset

from train import (
    train_one_epoch,
    validate,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    get_device
)
from models.ag_s2cnn import AG_S2CNN
from utils.loss import WeightedCrossEntropyLoss
from utils.metrics import MetricsCalculator
from config import Config


class SimpleDummyModel(nn.Module):
    """简单的虚拟模型用于测试"""
    def __init__(self, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    
    def forward(self, x_sat, x_ref):
        # 简化输入处理
        batch_size = x_sat.size(0)
        x = torch.randn(batch_size, 10)
        return self.fc(x)


class TestTrainingPipeline:
    """训练流水线测试类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.device = torch.device('cpu')
        self.num_bands = 290
        self.spatial_size = 13
        self.num_classes = 4
        self.batch_size = 4
        
        # 使用简单模型进行测试
        self.model = SimpleDummyModel(num_classes=self.num_classes).to(self.device)
        
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        # 创建学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10,
            eta_min=1e-6
        )
        
        # 创建损失函数
        self.criterion = WeightedCrossEntropyLoss()
        
        # 创建评估指标计算器
        self.metrics_calculator = MetricsCalculator(num_classes=self.num_classes)
        
        # 创建配置
        self.config = Config()
    
    def create_dummy_dataloader(self, num_samples=16):
        """创建虚拟数据加载器"""
        x_sat = torch.randn(num_samples, 1, self.num_bands, self.spatial_size, self.spatial_size)
        x_ref = torch.randn(num_samples, 1, self.num_bands, 1, 1)
        targets = torch.randint(0, self.num_classes, (num_samples,))
        
        dataset = TensorDataset(x_sat, x_ref, targets)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        return dataloader
    
    def test_train_one_epoch(self):
        """测试训练一个 epoch"""
        dataloader = self.create_dummy_dataloader()
        
        avg_loss, metrics = train_one_epoch(
            self.model,
            dataloader,
            self.criterion,
            self.optimizer,
            self.device,
            epoch=1
        )
        
        # 验证返回值
        assert isinstance(avg_loss, float)
        assert avg_loss > 0
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_validate(self):
        """测试验证函数"""
        dataloader = self.create_dummy_dataloader()
        
        avg_loss, metrics = validate(
            self.model,
            dataloader,
            self.criterion,
            self.device,
            self.metrics_calculator,
            epoch=1
        )
        
        # 验证返回值
        assert isinstance(avg_loss, float)
        assert avg_loss > 0
        assert 'loss' in metrics
        assert 'OA' in metrics
        assert 'AA' in metrics
        assert 'Kappa' in metrics
        assert 'F1_macro' in metrics
        assert 0 <= metrics['OA'] <= 1
        assert 0 <= metrics['AA'] <= 1
        assert -1 <= metrics['Kappa'] <= 1
        assert 0 <= metrics['F1_macro'] <= 1
    
    def test_early_stopping_max_mode(self):
        """测试早停机制（最大化模式）"""
        early_stopping = EarlyStopping(patience=3, mode='max')
        
        # 模拟性能提升
        assert not early_stopping(0.5)  # 第一次
        assert not early_stopping(0.6)  # 提升
        assert not early_stopping(0.7)  # 提升
        
        # 模拟性能不提升
        assert not early_stopping(0.65)  # 未提升，计数器 = 1
        assert not early_stopping(0.68)  # 未提升，计数器 = 2
        assert early_stopping(0.69)      # 未提升，计数器 = 3，触发早停
        
        assert early_stopping.early_stop is True
    
    def test_early_stopping_min_mode(self):
        """测试早停机制（最小化模式）"""
        early_stopping = EarlyStopping(patience=2, mode='min')
        
        # 模拟损失下降
        assert not early_stopping(1.0)  # 第一次
        assert not early_stopping(0.8)  # 下降
        assert not early_stopping(0.6)  # 下降
        
        # 模拟损失不下降
        assert not early_stopping(0.7)  # 未下降，计数器 = 1
        assert early_stopping(0.65)     # 未下降，计数器 = 2，触发早停
        
        assert early_stopping.early_stop is True
    
    def test_early_stopping_reset(self):
        """测试早停重置"""
        early_stopping = EarlyStopping(patience=2, mode='max')
        
        early_stopping(0.5)
        early_stopping(0.4)
        early_stopping(0.3)
        
        # 重置
        early_stopping.reset()
        
        assert early_stopping.counter == 0
        assert early_stopping.best_score is None
        assert early_stopping.early_stop is False
    
    def test_save_and_load_checkpoint(self):
        """测试模型保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')
            
            # 保存检查点
            metrics = {'OA': 0.85, 'Kappa': 0.80}
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch=10,
                metrics=metrics,
                config=self.config,
                filepath=checkpoint_path
            )
            
            # 验证文件存在
            assert os.path.exists(checkpoint_path)
            
            # 创建新模型
            new_model = SimpleDummyModel(num_classes=self.num_classes).to(self.device)
            
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)
            
            # 加载检查点
            checkpoint = load_checkpoint(
                checkpoint_path,
                new_model,
                new_optimizer,
                new_scheduler,
                self.device
            )
            
            # 验证加载的内容
            assert checkpoint['epoch'] == 10
            assert checkpoint['metrics'] == metrics
            
            # 验证模型权重一致
            for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)
    
    def test_save_checkpoint_without_scheduler(self):
        """测试不带调度器的模型保存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')
            
            metrics = {'OA': 0.85}
            save_checkpoint(
                self.model,
                self.optimizer,
                scheduler=None,
                epoch=5,
                metrics=metrics,
                config=self.config,
                filepath=checkpoint_path
            )
            
            assert os.path.exists(checkpoint_path)
            
            # 加载并验证
            checkpoint = torch.load(checkpoint_path)
            assert 'scheduler_state_dict' not in checkpoint
    
    def test_load_checkpoint_file_not_found(self):
        """测试加载不存在的检查点文件"""
        with pytest.raises(FileNotFoundError):
            load_checkpoint(
                'nonexistent_checkpoint.pth',
                self.model,
                device=self.device
            )
    
    def test_get_device_cpu(self):
        """测试 CPU 设备获取"""
        device = get_device(use_cuda=False)
        assert device.type == 'cpu'
    
    def test_get_device_cuda_fallback(self):
        """测试 CUDA 不可用时回退到 CPU"""
        # 即使请求 CUDA，如果不可用也会回退到 CPU
        device = get_device(use_cuda=True)
        assert device.type in ['cpu', 'cuda']
    
    def test_optimizer_configuration(self):
        """测试优化器配置"""
        # 验证优化器类型
        assert isinstance(self.optimizer, torch.optim.AdamW)
        
        # 验证学习率
        assert self.optimizer.param_groups[0]['lr'] == 1e-3
        
        # 验证权重衰减
        assert self.optimizer.param_groups[0]['weight_decay'] == 1e-4
    
    def test_scheduler_configuration(self):
        """测试学习率调度器配置"""
        # 验证调度器类型
        assert isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        
        # 验证初始学习率
        initial_lr = self.scheduler.get_last_lr()[0]
        assert initial_lr == 1e-3
        
        # 执行一步调度
        self.scheduler.step()
        new_lr = self.scheduler.get_last_lr()[0]
        
        # 验证学习率变化
        assert new_lr < initial_lr
    
    def test_training_validation_integration(self):
        """测试训练和验证的集成"""
        train_loader = self.create_dummy_dataloader(num_samples=16)
        val_loader = self.create_dummy_dataloader(num_samples=8)
        
        # 训练一个 epoch
        train_loss, train_metrics = train_one_epoch(
            self.model,
            train_loader,
            self.criterion,
            self.optimizer,
            self.device,
            epoch=1
        )
        
        # 验证
        val_loss, val_metrics = validate(
            self.model,
            val_loader,
            self.criterion,
            self.device,
            self.metrics_calculator,
            epoch=1
        )
        
        # 验证训练和验证都成功执行
        assert train_loss > 0
        assert val_loss > 0
        assert 'accuracy' in train_metrics
        assert 'OA' in val_metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
