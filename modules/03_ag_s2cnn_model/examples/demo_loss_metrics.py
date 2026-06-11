"""
损失函数和评估指标使用示例

演示如何使用 WeightedCrossEntropyLoss 和 MetricsCalculator
"""

import torch
import numpy as np
from models.ag_s2cnn import AG_S2CNN
from utils.loss import WeightedCrossEntropyLoss
from utils.metrics import MetricsCalculator


def demo_weighted_loss():
    """演示加权交叉熵损失函数的使用"""
    print("="*60)
    print("演示 1: 加权交叉熵损失函数")
    print("="*60)
    
    # 模拟不平衡的训练集标签分布
    # 类别 0: 100 个样本
    # 类别 1: 50 个样本
    # 类别 2: 20 个样本
    # 类别 3: 200 个样本
    train_labels = np.concatenate([
        np.zeros(100, dtype=int),
        np.ones(50, dtype=int),
        np.full(20, 2, dtype=int),
        np.full(200, 3, dtype=int)
    ])
    
    print(f"\n训练集标签分布:")
    for i in range(4):
        count = (train_labels == i).sum()
        print(f"  类别 {i}: {count} 个样本")
    
    # 计算类别权重
    weights = WeightedCrossEntropyLoss.calculate_class_weights(train_labels, num_classes=4)
    print(f"\n计算得到的类别权重:")
    for i in range(4):
        print(f"  类别 {i}: {weights[i]:.4f}")
    
    # 创建加权损失函数
    criterion = WeightedCrossEntropyLoss(class_weights=weights)
    
    # 模拟模型输出和目标
    batch_size = 8
    predictions = torch.randn(batch_size, 4)
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    
    # 计算损失
    loss = criterion(predictions, targets)
    print(f"\n加权交叉熵损失: {loss.item():.4f}")
    
    # 对比：不使用权重的损失
    criterion_unweighted = WeightedCrossEntropyLoss()
    loss_unweighted = criterion_unweighted(predictions, targets)
    print(f"标准交叉熵损失: {loss_unweighted.item():.4f}")
    print()


def demo_metrics_calculator():
    """演示评估指标计算器的使用"""
    print("="*60)
    print("演示 2: 评估指标计算器")
    print("="*60)
    
    # 模拟预测结果
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 4, size=n_samples)
    
    # 模拟 85% 准确率的预测
    y_pred = y_true.copy()
    n_errors = int(n_samples * 0.15)
    error_indices = np.random.choice(n_samples, n_errors, replace=False)
    for idx in error_indices:
        # 随机选择一个不同的类别
        wrong_class = np.random.choice([c for c in range(4) if c != y_true[idx]])
        y_pred[idx] = wrong_class
    
    # 创建评估指标计算器
    calculator = MetricsCalculator(num_classes=4)
    
    # 计算所有指标
    metrics = calculator.calculate_all_metrics(y_true, y_pred)
    
    # 打印指标报告
    class_names = ['锂辉石伟晶岩', '贫矿伟晶岩', '围岩', '背景']
    calculator.print_metrics(metrics, class_names)
    
    # 打印混淆矩阵
    calculator.print_confusion_matrix(metrics, class_names)


def demo_model_integration():
    """演示与模型的集成使用"""
    print("="*60)
    print("演示 3: 与 AG-S²CNN 模型集成")
    print("="*60)
    
    # 创建模型
    print("\n创建 AG-S²CNN 模型...")
    model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
    model.eval()
    
    # 创建输入数据
    batch_size = 4
    x_sat = torch.randn(batch_size, 1, 290, 13, 13)
    x_ref = torch.randn(batch_size, 1, 290, 1, 1)
    
    print(f"输入数据维度:")
    print(f"  卫星流: {x_sat.shape}")
    print(f"  地面流: {x_ref.shape}")
    
    # 前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        outputs = model(x_sat, x_ref)
    
    print(f"模型输出维度: {outputs.shape}")
    
    # 获取预测结果
    predictions = torch.argmax(outputs, dim=1)
    probabilities = torch.softmax(outputs, dim=1)
    
    print(f"\n预测结果:")
    for i in range(batch_size):
        pred_class = predictions[i].item()
        pred_prob = probabilities[i, pred_class].item()
        print(f"  样本 {i}: 类别 {pred_class} (置信度: {pred_prob:.4f})")
    
    # 计算损失
    targets = torch.tensor([0, 1, 2, 3])
    criterion = WeightedCrossEntropyLoss()
    loss = criterion(outputs, targets)
    
    print(f"\n交叉熵损失: {loss.item():.4f}")
    
    # 计算评估指标
    y_true = targets.numpy()
    y_pred = predictions.numpy()
    
    calculator = MetricsCalculator(num_classes=4)
    metrics = calculator.calculate_all_metrics(y_true, y_pred)
    
    print(f"\n评估指标:")
    print(f"  Overall Accuracy: {metrics['OA']:.4f}")
    print(f"  Average Accuracy: {metrics['AA']:.4f}")
    print(f"  Kappa 系数: {metrics['Kappa']:.4f}")
    print()


if __name__ == "__main__":
    # 运行所有演示
    demo_weighted_loss()
    demo_metrics_calculator()
    demo_model_integration()
    
    print("="*60)
    print("演示完成！")
    print("="*60)
