"""
损失函数模块

包含加权交叉熵损失函数，用于处理类别不平衡问题。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BinaryFocalLoss(nn.Module):
    """
    二分类 Focal Loss
    
    用于解决正负样本极度不平衡的问题（如找矿中背景像素远多于富矿像素）。
    
    参数:
        alpha (float): 正样本的权重系数，默认 0.75
        gamma (float): 难易样本的调节系数，默认 2.0
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算损失
        
        参数:
            inputs (Tensor): 模型输出 logits (Batch, 1) 或 (Batch,)
            targets (Tensor): 真实标签 0或1 (Batch,)
        """
        # 确保输入形状一致
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        # 使用更稳定的实现
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算概率 p
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Focal Loss 权重计算
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha 权重应用
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 最终损失
        loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失函数
    
    用于解决类别不平衡问题，对稀有类别赋予更高的权重。
    
    参数:
        class_weights (Tensor, optional): 类别权重 (num_classes,)
            如果为 None，则使用标准交叉熵损失
    
    示例:
        >>> # 计算类别权重
        >>> labels = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
        >>> weights = WeightedCrossEntropyLoss.calculate_class_weights(labels, num_classes=4)
        >>> 
        >>> # 创建损失函数
        >>> criterion = WeightedCrossEntropyLoss(class_weights=weights)
        >>> 
        >>> # 计算损失
        >>> predictions = torch.randn(10, 4)
        >>> targets = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
        >>> loss = criterion(predictions, targets)
    """
    
    def __init__(self, class_weights=None, label_smoothing: float = 0.0):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.label_smoothing = float(label_smoothing) if label_smoothing is not None else 0.0
        
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=self.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
    
    def forward(self, predictions, targets):
        """
        计算加权交叉熵损失
        
        参数:
            predictions (Tensor): 模型输出 logits (Batch, num_classes)
            targets (Tensor): 真实标签 (Batch,)
        
        返回:
            Tensor: 标量损失值
        """
        return self.criterion(predictions, targets)
    
    @staticmethod
    def calculate_class_weights(labels, num_classes):
        """
        根据训练集类别分布自动计算权重
        
        使用公式: w_c = N_total / (k × N_c)
        其中:
            - N_total: 总样本数
            - k: 类别数
            - N_c: 类别 c 的样本数
        
        对于零样本类别，权重设为 0。
        
        参数:
            labels (ndarray or list): 所有训练标签
            num_classes (int): 类别数
        
        返回:
            Tensor: 类别权重 (num_classes,)
        
        示例:
            >>> labels = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
            >>> weights = WeightedCrossEntropyLoss.calculate_class_weights(labels, num_classes=4)
            >>> print(weights)
            tensor([1.2500, 0.8333, 2.5000, 0.6250])
        """
        # 转换为 numpy 数组
        if isinstance(labels, list):
            labels = np.array(labels)
        
        # 统计每个类别的样本数
        class_counts = np.bincount(labels, minlength=num_classes)
        total_samples = len(labels)
        
        # 计算权重: w_c = N_total / (k * N_c)
        # 对于零样本类别，权重设为 0
        weights = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            if class_counts[c] > 0:
                weights[c] = total_samples / (num_classes * class_counts[c])
            else:
                weights[c] = 0.0
        
        # 归一化权重，使其和为 num_classes
        if weights.sum() > 0:
            weights = weights / weights.sum() * num_classes
        
        return torch.FloatTensor(weights)
