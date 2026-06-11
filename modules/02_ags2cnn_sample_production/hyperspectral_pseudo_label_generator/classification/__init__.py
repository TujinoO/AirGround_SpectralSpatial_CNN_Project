"""
Classification module for percentile thresholding and winner-takes-all classification.
用于百分位数阈值和赢者通吃分类的分类模块。
"""

from .thresholder import PercentileThresholder
from .classifier import WinnerTakesAllClassifier

__all__ = [
    "PercentileThresholder",
    "WinnerTakesAllClassifier",
]
