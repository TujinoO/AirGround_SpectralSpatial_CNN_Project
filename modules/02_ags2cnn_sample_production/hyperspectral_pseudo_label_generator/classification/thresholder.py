"""
Percentile thresholder module for SID scores.
用于SID分数的百分位数阈值器模块。
"""

import numpy as np
from typing import Dict, Optional
from ..config import ProcessingConfig


class PercentileThresholder:
    """
    Applies percentile-based thresholding to SID scores.
    对SID分数应用基于百分位数的阈值。
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize thresholder.
        初始化阈值器。
        
        Args:
            config: Processing configuration (处理配置)
        """
        self.config = config
        self.thresholds: Optional[Dict[int, float]] = None
    
    def compute_thresholds(self, sid_scores: np.ndarray) -> Dict[int, float]:
        """
        Compute percentile thresholds for each class.
        计算每个类别的百分位数阈值。
        
        Args:
            sid_scores: SID scores (H, W, num_classes)
                       SID分数
                       
        Returns:
            Dictionary mapping class index to threshold value
            将类别索引映射到阈值的字典
        """
        H, W, num_classes = sid_scores.shape
        thresholds: Dict[int, float] = {}
        
        for class_id in range(num_classes):
            # Get SID scores for this class
            # 获取此类别的SID分数
            class_scores = sid_scores[:, :, class_id].flatten()
            
            # Determine percentile based on class type
            # 根据类别类型确定百分位数
            if num_classes == 3:
                if class_id == 0:
                    percentile = self.config.ore_percentile
                elif class_id == 1:
                    percentile = self.config.poor_percentile
                else:
                    percentile = self.config.wall_percentile
            else:
                if class_id in [0, 1, 2]:  # Ore classes (矿石类别)
                    percentile = self.config.ore_percentile
                else:  # Non-ore classes (非矿石类别)
                    percentile = self.config.non_ore_percentile
            
            # Compute threshold (lower SID = more similar)
            # 计算阈值 (较低的SID = 更相似)
            threshold = np.percentile(class_scores, percentile)
            thresholds[class_id] = float(threshold)
        
        self.thresholds = thresholds
        return thresholds
    
    def apply_thresholds(self, sid_scores: np.ndarray) -> np.ndarray:
        """
        Apply thresholds to create binary masks for each class.
        应用阈值为每个类别创建二进制掩码。
        
        Args:
            sid_scores: SID scores (H, W, num_classes)
                       SID分数
                       
        Returns:
            Binary masks (H, W, num_classes) where True indicates passing threshold
            二进制掩码 (H, W, num_classes)，其中True表示通过阈值
        """
        if self.thresholds is None:
            raise RuntimeError("Must compute thresholds first")
        
        H, W, num_classes = sid_scores.shape
        masks = np.zeros((H, W, num_classes), dtype=bool)
        
        for class_id in range(num_classes):
            # Pixels with SID below threshold pass
            # SID低于阈值的像素通过
            masks[:, :, class_id] = sid_scores[:, :, class_id] <= self.thresholds[class_id]
        
        return masks
