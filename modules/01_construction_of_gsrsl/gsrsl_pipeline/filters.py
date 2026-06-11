"""
Spectral filtering module for GSRSL pipeline.
GSRSL管道的光谱滤波模块

This module provides functions for denoising and smoothing spectral data
while preserving important absorption features.
该模块提供用于去噪和平滑光谱数据的函数，同时保留重要的吸收特征。
"""

import logging
import numpy as np
from scipy.signal import savgol_filter
from typing import Optional

from gsrsl_pipeline.data_models import GroundSpectrum

logger = logging.getLogger(__name__)


def apply_savgol_filter(
    spectrum: GroundSpectrum,
    window_length: int = 15,
    polyorder: int = 3
) -> GroundSpectrum:
    """
    Apply Savitzky-Golay filter to denoise reflectance data.
    应用Savitzky-Golay滤波器对反射率数据进行去噪
    
    The Savitzky-Golay filter is a digital filter that smooths data by fitting
    successive sub-sets of adjacent data points with a low-degree polynomial
    using least-squares. This method is particularly effective for spectral data
    as it preserves absorption features while reducing noise.
    
    Savitzky-Golay滤波器是一种数字滤波器，通过使用最小二乘法用低阶多项式
    拟合相邻数据点的连续子集来平滑数据。该方法对光谱数据特别有效，
    因为它在降低噪声的同时保留吸收特征。
    
    Args:
        spectrum: GroundSpectrum object with raw reflectance data
                 包含原始反射率数据的GroundSpectrum对象
        
        window_length: Length of the filter window (must be odd and >= polyorder + 2)
                      滤波器窗口的长度（必须是奇数且 >= polyorder + 2）
                      Default: 15 (balances noise reduction with feature preservation
                                   for 1nm resolution data)
                      默认值：15（平衡1nm分辨率数据的噪声降低和特征保留）
        
        polyorder: Order of the polynomial used to fit the samples
                  用于拟合样本的多项式阶数
                  Default: 3 (cubic polynomial fits absorption features well
                             without overfitting)
                  默认值：3（三次多项式很好地拟合吸收特征而不会过拟合）
    
    Returns:
        New GroundSpectrum object with filtered reflectance values
        包含滤波后反射率值的新GroundSpectrum对象
        
        The wavelength array is preserved unchanged.
        波长数组保持不变。
        
        All other attributes (class_id, filename, is_anomalous) are copied
        from the input spectrum.
        所有其他属性（class_id、filename、is_anomalous）从输入光谱复制。
    
    Raises:
        ValueError: If spectrum has fewer data points than window_length
                   如果光谱的数据点少于window_length
        
        ValueError: If window_length is not odd or is too small
                   如果window_length不是奇数或太小
        
        ValueError: If polyorder >= window_length
                   如果polyorder >= window_length
    
    Example:
        >>> import numpy as np
        >>> from gsrsl_pipeline.data_models import GroundSpectrum
        >>> 
        >>> # Create a noisy spectrum
        >>> wavelengths = np.linspace(350, 2500, 2151, dtype=np.float64)
        >>> reflectances = np.sin(wavelengths / 100) * 0.3 + 0.5
        >>> reflectances += np.random.normal(0, 0.01, len(wavelengths))
        >>> 
        >>> spectrum = GroundSpectrum(
        ...     wavelengths=wavelengths,
        ...     reflectances=reflectances,
        ...     class_id=0,
        ...     filename="noisy_sample.csv",
        ...     is_anomalous=False
        ... )
        >>> 
        >>> # Apply filter
        >>> filtered = apply_savgol_filter(spectrum)
        >>> 
        >>> # Verify wavelengths unchanged
        >>> assert np.array_equal(filtered.wavelengths, spectrum.wavelengths)
        >>> 
        >>> # Verify reflectances are smoothed
        >>> assert np.std(filtered.reflectances) < np.std(spectrum.reflectances)
    
    References:
        Savitzky, A., & Golay, M. J. (1964). Smoothing and differentiation of
        data by simplified least squares procedures. Analytical chemistry, 36(8),
        1627-1639.
        
    Validates:
        Requirements 4.1, 4.2, 4.3, 4.4
    """
    # Validate window_length
    if window_length % 2 == 0:
        raise ValueError(
            f"window_length must be odd, got {window_length}. "
            f"Try using {window_length + 1} or {window_length - 1}."
        )
    
    if window_length < polyorder + 2:
        raise ValueError(
            f"window_length ({window_length}) must be >= polyorder + 2 ({polyorder + 2}). "
            f"Either increase window_length or decrease polyorder."
        )
    
    # Validate spectrum has enough data points
    n_points = len(spectrum)
    if n_points < window_length:
        raise ValueError(
            f"Spectrum has {n_points} data points but Savitzky-Golay filter "
            f"requires at least {window_length} points (window_length={window_length}). "
            f"Cannot apply filter to this spectrum. "
            f"Consider using a smaller window_length or obtaining higher resolution data."
        )
    
    # Log filtering operation
    logger.info(
        f"Applying Savitzky-Golay filter to {spectrum.filename}: "
        f"window_length={window_length}, polyorder={polyorder}, "
        f"n_points={n_points}"
    )
    
    # Apply Savitzky-Golay filter to reflectances
    # The filter preserves wavelengths and only smooths reflectances
    try:
        filtered_reflectances = savgol_filter(
            spectrum.reflectances,
            window_length=window_length,
            polyorder=polyorder,
            mode='interp'  # Use interpolation at boundaries
        )
    except Exception as e:
        logger.error(
            f"Failed to apply Savitzky-Golay filter to {spectrum.filename}: {e}"
        )
        raise
    
    # Ensure filtered reflectances remain in valid range [0.0, 1.0]
    # Clip values if necessary (can happen due to polynomial fitting at edges)
    original_min = filtered_reflectances.min()
    original_max = filtered_reflectances.max()
    
    filtered_reflectances = np.clip(filtered_reflectances, 0.0, 1.0)
    
    # Log if clipping was necessary
    if original_min < 0.0 or original_max > 1.0:
        logger.warning(
            f"Filtered reflectances for {spectrum.filename} were outside valid range "
            f"[{original_min:.6f}, {original_max:.6f}] and were clipped to [0.0, 1.0]. "
            f"This is normal for edge effects in polynomial fitting."
        )
    
    # Create new GroundSpectrum with filtered data
    # Preserve wavelengths unchanged (Requirement 4.2)
    # Copy other attributes from original spectrum
    filtered_spectrum = GroundSpectrum(
        wavelengths=spectrum.wavelengths.copy(),  # Preserve unchanged
        reflectances=filtered_reflectances.astype(np.float64),  # Filtered values
        class_id=spectrum.class_id,  # Copy from original
        filename=spectrum.filename,  # Copy from original
        is_anomalous=spectrum.is_anomalous  # Copy from original
    )
    
    # Log success
    logger.debug(
        f"Successfully filtered {spectrum.filename}: "
        f"reflectance range [{filtered_reflectances.min():.6f}, "
        f"{filtered_reflectances.max():.6f}]"
    )
    
    return filtered_spectrum
