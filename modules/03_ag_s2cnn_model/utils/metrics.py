"""
评估指标模块

包含多维度评估指标计算器，用于全面衡量模型性能。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score
)


class MetricsCalculator:
    """
    评估指标计算器
    
    功能:
        - Overall Accuracy (OA): 总体精度
        - Average Accuracy (AA): 平均精度（每个类别召回率的平均）
        - Kappa 系数: Cohen's Kappa 系数
        - F1-Score: 每个类别的 F1 分数
        - 混淆矩阵: 预测结果的混淆矩阵
        - 精确率和召回率: 每个类别的精确率和召回率
    
    参数:
        num_classes (int): 分类类别数
    
    示例:
        >>> calculator = MetricsCalculator(num_classes=4)
        >>> y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        >>> y_pred = np.array([0, 1, 2, 3, 0, 2, 2, 3])
        >>> metrics = calculator.calculate_all_metrics(y_true, y_pred)
        >>> print(f"OA: {metrics['OA']:.4f}")
        >>> print(f"Kappa: {metrics['Kappa']:.4f}")
    """
    
    def __init__(self, num_classes):
        # 如果模型输出1个通道(二分类)，我们在评估时其实是有0和1两个类别
        self.num_classes = 2 if num_classes == 1 else num_classes
    
    def calculate_all_metrics(self, y_true, y_pred, y_prob=None):
        """
        计算所有评估指标
        
        参数:
            y_true (ndarray): 真实标签 (N,)
            y_pred (ndarray): 预测标签 (N,)
            y_prob (ndarray, optional): 预测概率 (N,) 用于计算 AUC
        
        返回:
            dict: 包含所有指标的字典
                - 'OA': Overall Accuracy
                - 'AA': Average Accuracy
                - 'Kappa': Cohen's Kappa 系数
                - 'F1_per_class': 每个类别的 F1-Score (ndarray)
                - 'F1_macro': 宏平均 F1-Score
                - 'Precision_per_class': 每个类别的精确率 (ndarray)
                - 'Recall_per_class': 每个类别的召回率 (ndarray)
                - 'confusion_matrix': 混淆矩阵 (num_classes, num_classes)
                - 'AUC': AUC (仅二分类时且提供y_prob时返回)
        """
        metrics = {}
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        metrics['confusion_matrix'] = cm
        
        # Overall Accuracy (OA)
        metrics['OA'] = accuracy_score(y_true, y_pred)
        
        # Average Accuracy (AA) - 每个类别召回率的平均
        recalls = []
        for i in range(self.num_classes):
            if cm[i].sum() > 0:
                recall = cm[i, i] / cm[i].sum()
                recalls.append(recall)
        metrics['AA'] = np.mean(recalls) if recalls else 0.0
        
        # Kappa 系数
        metrics['Kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # F1-Score (每个类别)
        f1_scores = f1_score(
            y_true, y_pred,
            average=None,
            labels=range(self.num_classes),
            zero_division=0
        )
        metrics['F1_per_class'] = f1_scores
        metrics['F1_macro'] = np.mean(f1_scores)
        
        # 精确率 (每个类别)
        precision = precision_score(
            y_true, y_pred,
            average=None,
            labels=range(self.num_classes),
            zero_division=0
        )
        metrics['Precision_per_class'] = precision
        
        # 召回率 (每个类别)
        recall = recall_score(
            y_true, y_pred,
            average=None,
            labels=range(self.num_classes),
            zero_division=0
        )
        metrics['Recall_per_class'] = recall
        
        # AUC 计算 (二分类特有)
        if self.num_classes == 2 and y_prob is not None:
            from sklearn.metrics import roc_auc_score
            try:
                metrics['AUC'] = roc_auc_score(y_true, y_prob)
            except Exception:
                metrics['AUC'] = 0.0
                
        return metrics
    
    def print_metrics(self, metrics, class_names=None):
        """
        打印格式化的指标报告
        
        参数:
            metrics (dict): calculate_all_metrics() 返回的指标字典
            class_names (list, optional): 类别名称列表
                如果为 None，使用 "Class 0", "Class 1" 等默认名称
        
        示例:
            >>> calculator = MetricsCalculator(num_classes=4)
            >>> metrics = calculator.calculate_all_metrics(y_true, y_pred)
            >>> class_names = ['锂辉石伟晶岩', '贫矿伟晶岩', '围岩', '背景']
            >>> calculator.print_metrics(metrics, class_names)
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        print("\n" + "="*60)
        print("评估指标报告")
        print("="*60)
        print(f"Overall Accuracy (OA): {metrics['OA']:.4f}")
        print(f"Average Accuracy (AA): {metrics['AA']:.4f}")
        print(f"Kappa 系数: {metrics['Kappa']:.4f}")
        print(f"F1-Score (Macro): {metrics['F1_macro']:.4f}")
        print("\n每个类别的详细指标:")
        print("-"*60)
        print(f"{'类别':<20} {'精确率':<12} {'召回率':<12} {'F1-Score':<12}")
        print("-"*60)
        
        for i, name in enumerate(class_names):
            print(f"{name:<20} "
                  f"{metrics['Precision_per_class'][i]:<12.4f} "
                  f"{metrics['Recall_per_class'][i]:<12.4f} "
                  f"{metrics['F1_per_class'][i]:<12.4f}")
        
        print("="*60 + "\n")
    
    def print_confusion_matrix(self, metrics, class_names=None):
        """
        打印格式化的混淆矩阵
        
        参数:
            metrics (dict): calculate_all_metrics() 返回的指标字典
            class_names (list, optional): 类别名称列表
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        cm = metrics['confusion_matrix']
        
        print("\n混淆矩阵:")
        print("-"*60)
        
        # 打印表头
        header = "真实\\预测".ljust(15)
        for name in class_names:
            header += f"{name[:10]:<12}"
        print(header)
        print("-"*60)
        
        # 打印每一行
        for i, name in enumerate(class_names):
            row = f"{name[:15]:<15}"
            for j in range(self.num_classes):
                row += f"{cm[i, j]:<12}"
            print(row)
        
        print("-"*60 + "\n")
