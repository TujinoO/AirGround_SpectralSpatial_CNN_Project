"""
损失函数和评估指标集成测试

测试损失函数和评估指标与模型的集成
"""

import pytest
import torch
import numpy as np
from models.ag_s2cnn import AG_S2CNN
from utils.loss import WeightedCrossEntropyLoss
from utils.metrics import MetricsCalculator


def test_loss_with_model_output():
    """测试损失函数与模型输出的集成"""
    # 创建模型
    model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
    model.eval()
    
    # 创建输入数据
    batch_size = 4
    x_sat = torch.randn(batch_size, 1, 290, 13, 13)
    x_ref = torch.randn(batch_size, 1, 290, 1, 1)
    targets = torch.tensor([0, 1, 2, 3])
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x_sat, x_ref)
    
    # 计算损失
    criterion = WeightedCrossEntropyLoss()
    loss = criterion(outputs, targets)
    
    # 验证损失
    assert loss.dim() == 0, "损失应该是标量"
    assert loss.item() >= 0, "损失应该非负"
    assert not torch.isnan(loss), "损失不应该是 NaN"
    assert not torch.isinf(loss), "损失不应该是 Inf"


def test_weighted_loss_with_model():
    """测试加权损失函数与模型的集成"""
    # 创建模型
    model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
    model.eval()
    
    # 创建不平衡的标签分布
    labels = np.array([0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3])
    weights = WeightedCrossEntropyLoss.calculate_class_weights(labels, num_classes=4)
    
    # 创建加权损失函数
    criterion = WeightedCrossEntropyLoss(class_weights=weights)
    
    # 创建输入数据
    batch_size = 4
    x_sat = torch.randn(batch_size, 1, 290, 13, 13)
    x_ref = torch.randn(batch_size, 1, 290, 1, 1)
    targets = torch.tensor([0, 1, 2, 3])
    
    # 前向传播和损失计算
    with torch.no_grad():
        outputs = model(x_sat, x_ref)
    
    loss = criterion(outputs, targets)
    
    # 验证损失
    assert loss.dim() == 0, "损失应该是标量"
    assert loss.item() >= 0, "损失应该非负"


def test_metrics_with_model_predictions():
    """测试评估指标与模型预测的集成"""
    # 创建模型
    model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
    model.eval()
    
    # 创建输入数据
    batch_size = 20
    x_sat = torch.randn(batch_size, 1, 290, 13, 13)
    x_ref = torch.randn(batch_size, 1, 290, 1, 1)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x_sat, x_ref)
    
    # 获取预测结果
    predictions = torch.argmax(outputs, dim=1).numpy()
    
    # 创建真实标签
    y_true = np.random.randint(0, 4, size=batch_size)
    
    # 计算评估指标
    calculator = MetricsCalculator(num_classes=4)
    metrics = calculator.calculate_all_metrics(y_true, predictions)
    
    # 验证指标
    assert 0 <= metrics['OA'] <= 1, "OA 应该在 [0, 1] 范围内"
    assert 0 <= metrics['AA'] <= 1, "AA 应该在 [0, 1] 范围内"
    assert -1 <= metrics['Kappa'] <= 1, "Kappa 应该在 [-1, 1] 范围内"
    assert metrics['confusion_matrix'].shape == (4, 4), "混淆矩阵维度应该为 (4, 4)"


def test_training_loop_simulation():
    """模拟训练循环，测试损失函数和评估指标的完整流程"""
    # 创建模型
    model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
    
    # 创建损失函数
    criterion = WeightedCrossEntropyLoss()
    
    # 创建评估指标计算器
    calculator = MetricsCalculator(num_classes=4)
    
    # 模拟一个训练批次（使用小批量避免内存问题）
    batch_size = 2
    x_sat = torch.randn(batch_size, 1, 290, 13, 13)
    x_ref = torch.randn(batch_size, 1, 290, 1, 1)
    targets = torch.randint(0, 4, (batch_size,))
    
    # 前向传播（不进行实际训练，只测试流程）
    model.eval()
    with torch.no_grad():
        outputs = model(x_sat, x_ref)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 获取预测结果
        predictions = torch.argmax(outputs, dim=1).numpy()
        y_true = targets.numpy()
        
        # 计算评估指标
        metrics = calculator.calculate_all_metrics(y_true, predictions)
    
    # 验证步骤成功
    assert loss.item() >= 0, "损失应该非负"
    assert 0 <= metrics['OA'] <= 1, "OA 应该在 [0, 1] 范围内"
    assert metrics['confusion_matrix'].shape == (4, 4), "混淆矩阵维度应该为 (4, 4)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
