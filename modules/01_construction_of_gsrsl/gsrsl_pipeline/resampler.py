"""
Spectral resampler module for GSRSL pipeline.
GSRSL管道的光谱重采样器模块

This module implements Gaussian SRF (Spectral Response Function) convolution
to resample high-resolution ground spectra to satellite band resolution.
该模块实现高斯SRF（光谱响应函数）卷积，将高分辨率地面光谱重采样到卫星波段分辨率。

Physical Model:
物理模型：
- Satellite sensor spectral response is approximated as Gaussian distribution
  卫星传感器光谱响应近似为高斯分布
- FWHM (Full Width at Half Maximum) relates to Gaussian standard deviation
  FWHM（半高全宽）与高斯标准差的关系
- Mathematical basis: FWHM = 2√(2ln2)σ ≈ 2.355σ
  数学基础：FWHM = 2√(2ln2)σ ≈ 2.355σ
"""

import numpy as np
import logging
from typing import Optional
from .data_models import GroundSpectrum

# Configure logger
# 配置日志记录器
logger = logging.getLogger(__name__)


class SpectralResampler:
    """
    Resamples high-resolution ground spectra to satellite band resolution.
    将高分辨率地面光谱重采样到卫星波段分辨率
    
    This class implements physically-accurate spectral resampling using Gaussian
    spectral response functions (SRF). The resampling process convolves ground
    spectra with Gaussian weights to simulate how a satellite sensor integrates
    light across its spectral bands.
    
    该类使用高斯光谱响应函数（SRF）实现物理精确的光谱重采样。
    重采样过程将地面光谱与高斯权重进行卷积，以模拟卫星传感器如何在其光谱波段上积分光。
    
    Attributes:
        sat_wavelengths: Center wavelengths of satellite bands (nm), shape (297,)
                        卫星波段的中心波长（nm），形状(297,)
        sat_fwhms: FWHM values of satellite bands (nm), shape (297,)
                  卫星波段的FWHM值（nm），形状(297,)
        sat_sigmas: Gaussian standard deviations (nm), shape (297,)
                   高斯标准差（nm），形状(297,)
    
    Example:
        >>> # Load satellite specifications
        >>> sat_wavelengths, sat_fwhms = parse_metadata('gf5_metadata.txt')
        >>> 
        >>> # Create resampler
        >>> resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        >>> 
        >>> # Resample a ground spectrum
        >>> ground_spectrum = load_ground_spectrum('sample.csv', label_table)
        >>> resampled = resampler.resample(ground_spectrum)
        >>> print(resampled.shape)  # (297,)
    """
    
    def __init__(self, sat_wavelengths: np.ndarray, sat_fwhms: np.ndarray):
        """
        Initialize resampler with satellite band specifications.
        使用卫星波段规格初始化重采样器
        
        Args:
            sat_wavelengths: Center wavelengths of satellite bands, shape (297,)
                           卫星波段的中心波长，形状(297,)
                           Expected dtype: float64
                           预期数据类型：float64
            sat_fwhms: FWHM values of satellite bands, shape (297,)
                      卫星波段的FWHM值，形状(297,)
                      Expected dtype: float64
                      预期数据类型：float64
        
        Raises:
            TypeError: If inputs are not numpy arrays
            ValueError: If arrays don't have shape (297,) or contain invalid values
        """
        # Validate sat_wavelengths
        if not isinstance(sat_wavelengths, np.ndarray):
            raise TypeError("sat_wavelengths must be a numpy array")
        if sat_wavelengths.shape != (297,):
            raise ValueError(
                f"sat_wavelengths must have shape (297,), got {sat_wavelengths.shape}"
            )
        if not np.all(np.isfinite(sat_wavelengths)):
            raise ValueError("sat_wavelengths must contain only finite values")
        if not np.all(sat_wavelengths > 0):
            raise ValueError("sat_wavelengths must contain only positive values")
        
        # Validate sat_fwhms
        if not isinstance(sat_fwhms, np.ndarray):
            raise TypeError("sat_fwhms must be a numpy array")
        if sat_fwhms.shape != (297,):
            raise ValueError(
                f"sat_fwhms must have shape (297,), got {sat_fwhms.shape}"
            )
        if not np.all(np.isfinite(sat_fwhms)):
            raise ValueError("sat_fwhms must contain only finite values")
        if not np.all(sat_fwhms > 0):
            raise ValueError("sat_fwhms must contain only positive values")
        
        self.sat_wavelengths = sat_wavelengths
        self.sat_fwhms = sat_fwhms
        self.sat_sigmas = self._compute_sigmas(sat_fwhms)
        
        logger.info(
            f"SpectralResampler initialized with 297 satellite bands, "
            f"wavelength range: [{sat_wavelengths[0]:.2f}, {sat_wavelengths[-1]:.2f}] nm"
        )
    
    def _compute_sigmas(self, fwhms: np.ndarray) -> np.ndarray:
        """
        Convert FWHM to Gaussian standard deviation.
        将FWHM转换为高斯标准差
        
        The relationship between FWHM and standard deviation for a Gaussian
        distribution is derived from the definition of FWHM as the width at
        half the maximum value:
        
        高斯分布的FWHM和标准差之间的关系源于FWHM的定义，即最大值一半处的宽度：
        
        FWHM = 2√(2ln2) * σ
        
        Therefore: σ = FWHM / (2√(2ln2)) ≈ FWHM / 2.355
        因此：σ = FWHM / (2√(2ln2)) ≈ FWHM / 2.355
        
        Args:
            fwhms: FWHM values, shape (297,)
                  FWHM值，形状(297,)
        
        Returns:
            Array of sigma values, shape (297,), dtype float64
            sigma值数组，形状(297,)，数据类型float64
        
        Mathematical Derivation:
        数学推导：
        For a Gaussian f(x) = exp(-x²/(2σ²)), the maximum is at x=0 with f(0)=1.
        At half maximum, f(x) = 0.5, so:
        对于高斯函数f(x) = exp(-x²/(2σ²))，最大值在x=0处，f(0)=1。
        在半最大值处，f(x) = 0.5，因此：
        
        exp(-x²/(2σ²)) = 0.5
        -x²/(2σ²) = ln(0.5) = -ln(2)
        x² = 2σ²ln(2)
        x = σ√(2ln2)
        
        FWHM is the distance between the two half-maximum points:
        FWHM是两个半最大值点之间的距离：
        FWHM = 2x = 2σ√(2ln2)
        """
        # Compute the conversion factor: 2 * sqrt(2 * ln(2))
        # 计算转换因子：2 * sqrt(2 * ln(2))
        factor = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.355
        
        # Convert FWHM to sigma
        # 将FWHM转换为sigma
        sigmas = fwhms / factor
        
        return sigmas
    
    def _gaussian_weight(self, wavelength: float, center: float, sigma: float) -> float:
        """
        Compute Gaussian weight for a given wavelength.
        计算给定波长的高斯权重
        
        The Gaussian weight function describes how a sensor responds to different
        wavelengths. The weight is maximum at the band center and decreases
        exponentially with distance from the center.
        
        高斯权重函数描述传感器对不同波长的响应。
        权重在波段中心处最大，并随着与中心的距离呈指数衰减。
        
        Formula: w(λ) = exp(-(λ - λ_center)² / (2σ²))
        公式：w(λ) = exp(-(λ - λ_center)² / (2σ²))
        
        Args:
            wavelength: Wavelength to compute weight for (nm)
                       要计算权重的波长（nm）
            center: Center wavelength of satellite band (nm)
                   卫星波段的中心波长（nm）
            sigma: Standard deviation of Gaussian (nm)
                  高斯的标准差（nm）
        
        Returns:
            Weight value in range [0.0, 1.0]
            权重值，范围[0.0, 1.0]
            
        Physical Interpretation:
        物理解释：
        - At λ = λ_center: w = 1.0 (maximum sensitivity)
          在λ = λ_center处：w = 1.0（最大灵敏度）
        - At λ = λ_center ± σ: w ≈ 0.606 (60.6% sensitivity)
          在λ = λ_center ± σ处：w ≈ 0.606（60.6%灵敏度）
        - At λ = λ_center ± 3σ: w ≈ 0.011 (1.1% sensitivity)
          在λ = λ_center ± 3σ处：w ≈ 0.011（1.1%灵敏度）
        """
        # Compute squared distance from center
        # 计算与中心的平方距离
        distance_squared = (wavelength - center) ** 2
        
        # Compute Gaussian weight
        # 计算高斯权重
        weight = np.exp(-distance_squared / (2.0 * sigma ** 2))
        
        return weight
    
    def resample(self, spectrum: GroundSpectrum) -> np.ndarray:
        """
        Resample ground spectrum to satellite band resolution.
        将地面光谱重采样到卫星波段分辨率
        
        This method performs Gaussian SRF convolution to convert high-resolution
        ground spectra (typically 1nm resolution) to satellite band resolution
        (typically 4-8nm resolution). For each satellite band, the method:
        
        该方法执行高斯SRF卷积，将高分辨率地面光谱（通常为1nm分辨率）
        转换为卫星波段分辨率（通常为4-8nm分辨率）。对于每个卫星波段，该方法：
        
        1. Identifies overlapping ground wavelengths (within ±3σ of band center)
           识别重叠的地面波长（在波段中心的±3σ范围内）
        2. Computes Gaussian weights for each overlapping wavelength
           计算每个重叠波长的高斯权重
        3. Computes weighted average of ground reflectances
           计算地面反射率的加权平均值
        4. Normalizes by sum of weights to ensure energy conservation
           通过权重总和归一化以确保能量守恒
        
        Args:
            spectrum: GroundSpectrum object with denoised data
                     包含去噪数据的GroundSpectrum对象
        
        Returns:
            Resampled reflectance array, shape (297,), dtype float32
            重采样的反射率数组，形状(297,)，数据类型float32
            
            Note: Bands with no overlapping ground data will have NaN values
            注意：没有重叠地面数据的波段将具有NaN值
        
        Raises:
            TypeError: If spectrum is not a GroundSpectrum object
        
        Algorithm Details:
        算法细节：
        
        For each satellite band i:
        对于每个卫星波段i：
        
        1. Define integration range: [λ_center - 3σ, λ_center + 3σ]
           定义积分范围：[λ_center - 3σ, λ_center + 3σ]
           (±3σ covers 99.7% of Gaussian distribution)
           (±3σ覆盖99.7%的高斯分布)
        
        2. Find overlapping ground wavelengths: λ_j ∈ [λ_min, λ_max]
           查找重叠的地面波长：λ_j ∈ [λ_min, λ_max]
        
        3. Compute weights: w_j = exp(-(λ_j - λ_center)² / (2σ²))
           计算权重：w_j = exp(-(λ_j - λ_center)² / (2σ²))
        
        4. Compute weighted average: R_sat = Σ(R_j * w_j) / Σ(w_j)
           计算加权平均：R_sat = Σ(R_j * w_j) / Σ(w_j)
        
        Edge Cases:
        边缘情况：
        - No overlap: Set R_sat = NaN and log warning
          无重叠：设置R_sat = NaN并记录警告
        - Partial overlap: Use only available wavelengths (no extrapolation)
          部分重叠：仅使用可用波长（无外推）
        """
        # Validate input
        if not isinstance(spectrum, GroundSpectrum):
            raise TypeError(
                f"spectrum must be a GroundSpectrum object, got {type(spectrum)}"
            )
        
        # Initialize output array
        # 初始化输出数组
        resampled = np.zeros(297, dtype=np.float32)
        
        # Track statistics for logging
        # 跟踪统计信息以进行日志记录
        n_no_overlap = 0
        n_partial_overlap = 0
        
        # Resample each satellite band
        # 重采样每个卫星波段
        for i in range(297):
            center = self.sat_wavelengths[i]
            sigma = self.sat_sigmas[i]
            
            # Define integration range: ±3σ covers 99.7% of Gaussian
            # 定义积分范围：±3σ覆盖99.7%的高斯分布
            lambda_min = center - 3.0 * sigma
            lambda_max = center + 3.0 * sigma
            
            # Find overlapping ground wavelengths
            # 查找重叠的地面波长
            mask = (spectrum.wavelengths >= lambda_min) & (spectrum.wavelengths <= lambda_max)
            
            if not np.any(mask):
                # No overlap: set to NaN and log warning
                # 无重叠：设置为NaN并记录警告
                resampled[i] = np.nan
                n_no_overlap += 1
                logger.warning(
                    f"Band {i} (λ={center:.2f}nm, σ={sigma:.2f}nm) has no overlapping "
                    f"ground data in range [{lambda_min:.2f}, {lambda_max:.2f}] nm. "
                    f"Ground spectrum range: [{spectrum.wavelengths[0]:.2f}, "
                    f"{spectrum.wavelengths[-1]:.2f}] nm. Setting to NaN."
                )
                continue
            
            # Extract overlapping data
            # 提取重叠数据
            ground_wavelengths = spectrum.wavelengths[mask]
            ground_reflectances = spectrum.reflectances[mask]
            
            # Check for partial overlap (some wavelengths in ±3σ range are missing)
            # 检查部分重叠（±3σ范围内的某些波长缺失）
            expected_range = lambda_max - lambda_min
            actual_range = ground_wavelengths[-1] - ground_wavelengths[0]
            if actual_range < 0.9 * expected_range:
                n_partial_overlap += 1
                logger.debug(
                    f"Band {i} (λ={center:.2f}nm) has partial overlap: "
                    f"expected range {expected_range:.2f}nm, "
                    f"actual range {actual_range:.2f}nm"
                )
            
            # Compute weights for each ground wavelength
            # 计算每个地面波长的权重
            weights = np.array([
                self._gaussian_weight(wl, center, sigma)
                for wl in ground_wavelengths
            ])
            
            # Compute weighted average (normalized)
            # 计算加权平均（归一化）
            weighted_sum = np.sum(ground_reflectances * weights)
            weight_sum = np.sum(weights)
            
            resampled[i] = weighted_sum / weight_sum
        
        # Log summary statistics
        # 记录摘要统计信息
        if n_no_overlap > 0:
            logger.warning(
                f"Resampling complete: {n_no_overlap} bands ({n_no_overlap/297*100:.1f}%) "
                f"have no overlapping ground data (set to NaN)"
            )
        
        if n_partial_overlap > 0:
            logger.info(
                f"Resampling complete: {n_partial_overlap} bands ({n_partial_overlap/297*100:.1f}%) "
                f"have partial overlap (used available wavelengths only)"
            )
        
        if n_no_overlap == 0 and n_partial_overlap == 0:
            logger.info(
                f"Resampling complete: all 297 bands have full overlap with ground spectrum "
                f"(range: [{spectrum.wavelengths[0]:.2f}, {spectrum.wavelengths[-1]:.2f}] nm)"
            )
        
        return resampled
