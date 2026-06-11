"""
Class aggregator module for GSRSL pipeline.
GSRSL管道的类别聚合器模块

This module provides functionality to aggregate resampled spectra by lithology class,
computing mean reference spectra for each class.
该模块提供按岩性类别聚合重采样光谱的功能，计算每个类别的平均参考光谱。
"""

import logging
from typing import List, Tuple, Dict
import numpy as np

# Configure logger
# 配置日志记录器
logger = logging.getLogger(__name__)


def aggregate_by_class(
    resampled_spectra: List[Tuple[int, np.ndarray]]
) -> Dict[int, np.ndarray]:
    """
    Compute mean reference spectrum for each lithology class.
    计算每个岩性类别的平均参考光谱
    
    This function groups resampled spectra by their class_id and computes the
    arithmetic mean for each class. NaN values are handled gracefully using
    np.nanmean, which ignores NaN values when computing the mean. If all samples
    for a particular band are NaN, the result will be NaN for that band.
    
    该函数按class_id对重采样光谱进行分组，并计算每个类别的算术平均值。
    使用np.nanmean优雅地处理NaN值，在计算平均值时忽略NaN值。
    如果特定波段的所有样本都是NaN，则该波段的结果将是NaN。
    
    Args:
        resampled_spectra: List of (class_id, resampled_array) tuples
                          (class_id, resampled_array)元组的列表
                          - class_id: Integer in range [0, 4] representing lithology class
                                     范围[0, 4]内的整数，表示岩性类别
                          - resampled_array: numpy array of shape (297,) with dtype float32
                                            形状为(297,)、数据类型为float32的numpy数组
    
    Returns:
        Dictionary mapping class_id (0-4) to mean spectrum array (297,)
        将class_id(0-4)映射到平均光谱数组(297,)的字典
        
        Each mean spectrum has:
        - shape: (297,)
        - dtype: float32
        - values: reflectance in range [0.0, 1.0] or NaN
        
        每个平均光谱具有：
        - 形状：(297,)
        - 数据类型：float32
        - 值：范围[0.0, 1.0]内的反射率或NaN
    
    Raises:
        ValueError: If resampled_spectra is empty
                   如果resampled_spectra为空
        ValueError: If any class_id is not in range [0, 4]
                   如果任何class_id不在范围[0, 4]内
        ValueError: If no samples found for any class
                   如果任何类别没有找到样本
        TypeError: If resampled_spectra is not a list
                  如果resampled_spectra不是列表
        TypeError: If any resampled array is not a numpy array
                  如果任何重采样数组不是numpy数组
        ValueError: If any resampled array does not have shape (297,)
                   如果任何重采样数组的形状不是(297,)
    
    Example:
        >>> # Create sample resampled spectra
        >>> spectrum1 = (0, np.full(297, 0.3, dtype=np.float32))
        >>> spectrum2 = (0, np.full(297, 0.5, dtype=np.float32))
        >>> spectrum3 = (1, np.full(297, 0.7, dtype=np.float32))
        >>> 
        >>> resampled = [spectrum1, spectrum2, spectrum3]
        >>> class_means = aggregate_by_class(resampled)
        >>> 
        >>> # Class 0 mean should be 0.4 (average of 0.3 and 0.5)
        >>> assert np.allclose(class_means[0], 0.4)
        >>> # Class 1 mean should be 0.7
        >>> assert np.allclose(class_means[1], 0.7)
    
    Notes:
        - Uses np.nanmean to ignore NaN values in averaging
          使用np.nanmean在平均时忽略NaN值
        - If all samples for a band contain NaN, the mean will be NaN for that band
          如果波段的所有样本都包含NaN，则该波段的平均值将是NaN
        - Logs a warning if any class has > 10% NaN bands in final result
          如果任何类别在最终结果中有>10%的NaN波段，则记录警告
        - Each lithology class is processed independently
          每个岩性类别独立处理
    
    Lithology Classes:
    岩性类别：
        0: Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)
        1: Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)
        2: Mixed-type Rich Pegmatite (混合型富矿伟晶岩)
        3: Barren Pegmatite (贫矿伟晶岩)
        4: Wall Rock (围岩)
    """
    # Validate input type
    # 验证输入类型
    if not isinstance(resampled_spectra, list):
        raise TypeError(
            f"resampled_spectra must be a list, got {type(resampled_spectra).__name__}"
        )
    
    # Validate not empty
    # 验证非空
    if len(resampled_spectra) == 0:
        raise ValueError("resampled_spectra cannot be empty")
    
    # Initialize groups for each class
    # 为每个类别初始化分组
    class_groups = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    # Group spectra by class_id
    # 按class_id分组光谱
    for i, item in enumerate(resampled_spectra):
        # Validate tuple structure
        # 验证元组结构
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                f"Each item in resampled_spectra must be a tuple of (class_id, array), "
                f"got {type(item).__name__} at index {i}"
            )
        
        class_id, spectrum = item
        
        # Validate class_id
        # 验证class_id
        if not isinstance(class_id, (int, np.integer)):
            raise TypeError(
                f"class_id must be an integer, got {type(class_id).__name__} at index {i}"
            )
        
        if not (0 <= class_id <= 4):
            raise ValueError(
                f"class_id must be in range [0, 4], got {class_id} at index {i}"
            )
        
        # Validate spectrum array
        # 验证光谱数组
        if not isinstance(spectrum, np.ndarray):
            raise TypeError(
                f"Spectrum must be a numpy array, got {type(spectrum).__name__} at index {i}"
            )
        
        if spectrum.shape != (297,):
            raise ValueError(
                f"Spectrum must have shape (297,), got {spectrum.shape} at index {i}"
            )
        
        # Add to appropriate group
        # 添加到适当的组
        class_groups[class_id].append(spectrum)
    
    # Compute mean for each class
    # 计算每个类别的平均值
    class_means = {}
    
    # Class names for logging
    # 用于日志记录的类别名称
    class_names = {
        0: "Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)",
        1: "Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)",
        2: "Mixed-type Rich Pegmatite (混合型富矿伟晶岩)",
        3: "Barren Pegmatite (贫矿伟晶岩)",
        4: "Wall Rock (围岩)"
    }
    
    for class_id in range(5):
        # Check if class has samples
        # 检查类别是否有样本
        if len(class_groups[class_id]) == 0:
            raise ValueError(
                f"No samples found for class {class_id} ({class_names[class_id]}). "
                f"Each class must have at least one sample."
            )
        
        # Stack arrays and compute nanmean along axis 0
        # 堆叠数组并沿轴0计算nanmean
        stacked = np.stack(class_groups[class_id], axis=0)
        mean_spectrum = np.nanmean(stacked, axis=0).astype(np.float32)
        
        # Log information about the aggregation
        # 记录有关聚合的信息
        n_samples = len(class_groups[class_id])
        n_nan_bands = np.sum(np.isnan(mean_spectrum))
        nan_percentage = (n_nan_bands / 297.0) * 100.0
        
        logger.info(
            f"Class {class_id} ({class_names[class_id]}): "
            f"aggregated {n_samples} samples, "
            f"{n_nan_bands} NaN bands ({nan_percentage:.1f}%)"
        )
        
        # Warn if > 10% NaN bands
        # 如果>10%的NaN波段则警告
        if nan_percentage > 10.0:
            logger.warning(
                f"Class {class_id} ({class_names[class_id]}) has {nan_percentage:.1f}% NaN bands "
                f"in the mean spectrum. This may indicate insufficient wavelength coverage "
                f"in the ground spectra for this class."
            )
        
        class_means[class_id] = mean_spectrum
    
    logger.info(
        f"Successfully aggregated spectra for all 5 lithology classes. "
        f"Total samples processed: {len(resampled_spectra)}"
    )
    
    return class_means
