"""
Winner-takes-all classifier module with ambiguity handling.
带有歧义处理的赢者通吃分类器模块。
"""

import numpy as np
from typing import Optional
from ..config import ProcessingConfig


class WinnerTakesAllClassifier:
    """
    Implements winner-takes-all classification with ambiguity handling.
    实现带有歧义处理的赢者通吃分类。
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize classifier.
        初始化分类器。
        
        Args:
            config: Processing configuration (处理配置)
        """
        self.config = config
        self._num_classes: Optional[int] = None
    
    def classify(self, 
                 sid_scores: np.ndarray, 
                 masks: np.ndarray) -> np.ndarray:
        """
        Classify pixels using winner-takes-all with ambiguity detection.
        使用带有歧义检测的赢者通吃策略对像素进行分类。
        
        Args:
            sid_scores: SID scores (H, W, num_classes)
                       SID分数
            masks: Binary masks indicating which pixels pass thresholds
                  指示哪些像素通过阈值的二进制掩码
                  
        Returns:
            Pseudo-label map (H, W) with labels:
            伪标签图 (H, W)，标签为：
            - 0: Unlabeled/ambiguous (未标注/歧义)
            - 1: Rich Ore Pegmatite (富矿伟晶岩)
            - 2: Poor Ore Pegmatite (贫矿伟晶岩)
            - 3: Wall Rock (围岩)
        """
        H, W, num_classes = sid_scores.shape
        self._num_classes = num_classes
        
        # Initialize with label 0 (unlabeled)
        # 初始化为标签0 (未标注)
        pseudo_labels = np.zeros((H, W), dtype=np.uint8)
        
        for i in range(H):
            for j in range(W):
                # Get classes that pass threshold for this pixel
                # 获取此像素通过阈值的类别
                passing_classes = np.where(masks[i, j, :])[0]
                
                if len(passing_classes) == 0:
                    # No class passes threshold → label 0
                    # 没有类别通过阈值 → 标签0
                    continue
                
                elif len(passing_classes) == 1:
                    # Only one class passes → assign that label
                    # 只有一个类别通过 → 分配该标签
                    class_id = passing_classes[0]
                    pseudo_labels[i, j] = self._map_class_to_label(class_id)
                
                else:
                    # Multiple classes pass → check for ambiguity
                    # 多个类别通过 → 检查歧义
                    label = self._resolve_ambiguity(
                        sid_scores[i, j, :], 
                        passing_classes
                    )
                    pseudo_labels[i, j] = label
        
        return pseudo_labels
    
    def _resolve_ambiguity(self, 
                          pixel_scores: np.ndarray, 
                          passing_classes: np.ndarray) -> int:
        """
        Resolve ambiguity when multiple classes pass threshold.
        当多个类别通过阈值时解决歧义。
        
        Args:
            pixel_scores: SID scores for this pixel (num_classes,)
                         此像素的SID分数
            passing_classes: Indices of classes that passed threshold
                           通过阈值的类别索引
                           
        Returns:
            Label (0 if ambiguous, otherwise mapped class label)
            标签 (如果歧义则为0，否则为映射的类别标签)
        """
        # Get SID scores for passing classes
        # 获取通过类别的SID分数
        scores = pixel_scores[passing_classes]
        
        # Sort to find two lowest scores
        # 排序以找到两个最低分数
        sorted_indices = np.argsort(scores)
        lowest_score = scores[sorted_indices[0]]
        second_lowest_score = scores[sorted_indices[1]]
        
        # Check ambiguity
        # 检查歧义
        if second_lowest_score - lowest_score < self.config.ambiguity_threshold:
            # Too ambiguous → label 0
            # 太歧义 → 标签0
            return 0
        
        # Clear winner → assign label
        # 明确的赢家 → 分配标签
        winner_class = passing_classes[sorted_indices[0]]
        return self._map_class_to_label(winner_class)
    
    def _map_class_to_label(self, class_id: int) -> int:
        """
        Map internal class index to output label.
        将内部类别索引映射到输出标签。
        
        Args:
            class_id: Internal class index (0-4)
                     内部类别索引
                     
        Returns:
            Output label (1-3)
            输出标签
        """
        if self._num_classes is None:
            self._num_classes = 5

        if self._num_classes == 3:
            if class_id == 0:
                return 1
            if class_id == 1:
                return 2
            if class_id == 2:
                return 3
            raise ValueError(f"Invalid class_id for 3-class mode: {class_id}")

        if class_id in [0, 1, 2]:
            return 1
        if class_id == 3:
            return 2
        if class_id == 4:
            return 3
        raise ValueError(f"Invalid class_id: {class_id}")
