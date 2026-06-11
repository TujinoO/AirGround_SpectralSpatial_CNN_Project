"""
损失函数测试

测试 WeightedCrossEntropyLoss 的基本功能
"""

import pytest
import torch
import numpy as np
from utils.loss import WeightedCrossEntropyLoss


def test_weighted_cross_entropy_loss_basic():
    """测试基本的加权交叉熵损失计算"""
    # 创建简单的测试数据
    predictions = torch.randn(10, 4)
    targets = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
    
    # 不使用权重
    criterion = WeightedCrossEntropyLoss()
    loss = criterion(predictions, targets)
    
    # 验证损失是标量
    assert loss.dim() == 0, "损失应该是标量"
    assert loss.item() >= 0, "损失应该非负"


def test_calculate_class_weights():
    """测试类别权重计算"""
    labels = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
    num_classes = 4
    
    weights = WeightedCrossEntropyLoss.calculate_class_weights(labels, num_classes)
    
    # 验证权重维度
    assert weights.shape == (num_classes,), f"权重维度应为 ({num_classes},)"
    
    # 验证权重非负
    assert (weights >= 0).all(), "权重应该全部非负"
    
    # 验证权重和为 num_classes（归一化后）
    assert abs(weights.sum().item() - num_classes) < 1e-5, "权重和应该等于类别数"


def test_calculate_class_weights_with_zero_samples():
    """测试零样本类别的权重计算"""
    # 类别 2 没有样本
    labels = np.array([0, 0, 1, 1, 1, 3, 3, 3, 3])
    num_classes = 4
    
    weights = WeightedCrossEntropyLoss.calculate_class_weights(labels, num_classes)
    
    # 验证零样本类别的权重为 0
    assert weights[2].item() == 0.0, "零样本类别的权重应该为 0"
    
    # 验证其他类别的权重非零
    assert weights[0].item() > 0, "有样本的类别权重应该大于 0"
    assert weights[1].item() > 0, "有样本的类别权重应该大于 0"
    assert weights[3].item() > 0, "有样本的类别权重应该大于 0"


def test_weighted_loss_with_weights():
    """测试使用权重的损失计算"""
    labels = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
    num_classes = 4
    
    # 计算权重
    weights = WeightedCrossEntropyLoss.calculate_class_weights(labels, num_classes)
    
    # 创建加权损失函数
    criterion = WeightedCrossEntropyLoss(class_weights=weights)
    
    # 测试数据
    predictions = torch.randn(10, 4)
    targets = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
    
    loss = criterion(predictions, targets)
    
    # 验证损失是标量且非负
    assert loss.dim() == 0, "损失应该是标量"
    assert loss.item() >= 0, "损失应该非负"


def test_weighted_loss_backward():
    """测试损失函数的反向传播"""
    predictions = torch.randn(10, 4, requires_grad=True)
    targets = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
    
    criterion = WeightedCrossEntropyLoss()
    loss = criterion(predictions, targets)
    
    # 反向传播
    loss.backward()
    
    # 验证梯度已计算
    assert predictions.grad is not None, "梯度应该已计算"
    assert predictions.grad.shape == predictions.shape, "梯度维度应该与输入一致"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
