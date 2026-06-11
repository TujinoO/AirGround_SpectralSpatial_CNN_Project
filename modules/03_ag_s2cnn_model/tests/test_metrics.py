"""
评估指标测试

测试 MetricsCalculator 的基本功能
"""

import pytest
import numpy as np
from utils.metrics import MetricsCalculator


def test_metrics_calculator_basic():
    """测试基本的指标计算"""
    calculator = MetricsCalculator(num_classes=4)
    
    # 完美预测
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    
    metrics = calculator.calculate_all_metrics(y_true, y_pred)
    
    # 验证 OA 为 1.0
    assert abs(metrics['OA'] - 1.0) < 1e-6, "完美预测的 OA 应该为 1.0"
    
    # 验证 AA 为 1.0
    assert abs(metrics['AA'] - 1.0) < 1e-6, "完美预测的 AA 应该为 1.0"
    
    # 验证 Kappa 为 1.0
    assert abs(metrics['Kappa'] - 1.0) < 1e-6, "完美预测的 Kappa 应该为 1.0"
    
    # 验证 F1-Score 全为 1.0
    assert np.allclose(metrics['F1_per_class'], 1.0), "完美预测的 F1-Score 应该全为 1.0"


def test_metrics_calculator_imperfect():
    """测试非完美预测的指标计算"""
    calculator = MetricsCalculator(num_classes=4)
    
    # 有错误的预测
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 2, 2, 3])  # 第 5 个样本预测错误
    
    metrics = calculator.calculate_all_metrics(y_true, y_pred)
    
    # 验证 OA 在 [0, 1] 范围内
    assert 0 <= metrics['OA'] <= 1, "OA 应该在 [0, 1] 范围内"
    
    # 验证 AA 在 [0, 1] 范围内
    assert 0 <= metrics['AA'] <= 1, "AA 应该在 [0, 1] 范围内"
    
    # 验证 Kappa 在 [-1, 1] 范围内
    assert -1 <= metrics['Kappa'] <= 1, "Kappa 应该在 [-1, 1] 范围内"
    
    # 验证 F1-Score 在 [0, 1] 范围内
    assert np.all((metrics['F1_per_class'] >= 0) & (metrics['F1_per_class'] <= 1)), \
        "F1-Score 应该在 [0, 1] 范围内"


def test_confusion_matrix():
    """测试混淆矩阵生成"""
    calculator = MetricsCalculator(num_classes=4)
    
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 2, 2, 3])
    
    metrics = calculator.calculate_all_metrics(y_true, y_pred)
    cm = metrics['confusion_matrix']
    
    # 验证混淆矩阵维度
    assert cm.shape == (4, 4), "混淆矩阵维度应该为 (4, 4)"
    
    # 验证混淆矩阵非负
    assert (cm >= 0).all(), "混淆矩阵所有元素应该非负"
    
    # 验证混淆矩阵总和等于样本数
    assert cm.sum() == len(y_true), "混淆矩阵总和应该等于样本数"


def test_metrics_with_missing_class():
    """测试某些类别在测试集中不存在的情况"""
    calculator = MetricsCalculator(num_classes=4)
    
    # 类别 3 不存在
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    
    metrics = calculator.calculate_all_metrics(y_true, y_pred)
    
    # 验证指标仍然可以计算
    assert 0 <= metrics['OA'] <= 1, "OA 应该在 [0, 1] 范围内"
    assert 0 <= metrics['AA'] <= 1, "AA 应该在 [0, 1] 范围内"
    
    # 验证缺失类别的 F1-Score 为 0
    assert metrics['F1_per_class'][3] == 0.0, "缺失类别的 F1-Score 应该为 0"


def test_precision_recall():
    """测试精确率和召回率计算"""
    calculator = MetricsCalculator(num_classes=4)
    
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    
    metrics = calculator.calculate_all_metrics(y_true, y_pred)
    
    # 验证精确率和召回率维度
    assert metrics['Precision_per_class'].shape == (4,), "精确率维度应该为 (4,)"
    assert metrics['Recall_per_class'].shape == (4,), "召回率维度应该为 (4,)"
    
    # 完美预测的精确率和召回率应该全为 1.0
    assert np.allclose(metrics['Precision_per_class'], 1.0), "完美预测的精确率应该全为 1.0"
    assert np.allclose(metrics['Recall_per_class'], 1.0), "完美预测的召回率应该全为 1.0"


def test_print_metrics(capsys):
    """测试打印指标报告"""
    calculator = MetricsCalculator(num_classes=4)
    
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    
    metrics = calculator.calculate_all_metrics(y_true, y_pred)
    
    # 测试打印功能（不应该抛出异常）
    calculator.print_metrics(metrics)
    
    # 捕获输出
    captured = capsys.readouterr()
    
    # 验证输出包含关键信息
    assert "Overall Accuracy" in captured.out, "输出应该包含 Overall Accuracy"
    assert "Kappa" in captured.out, "输出应该包含 Kappa"
    assert "F1-Score" in captured.out, "输出应该包含 F1-Score"


def test_print_confusion_matrix(capsys):
    """测试打印混淆矩阵"""
    calculator = MetricsCalculator(num_classes=4)
    
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    
    metrics = calculator.calculate_all_metrics(y_true, y_pred)
    
    # 测试打印功能（不应该抛出异常）
    calculator.print_confusion_matrix(metrics)
    
    # 捕获输出
    captured = capsys.readouterr()
    
    # 验证输出包含混淆矩阵
    assert "混淆矩阵" in captured.out, "输出应该包含混淆矩阵标题"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
