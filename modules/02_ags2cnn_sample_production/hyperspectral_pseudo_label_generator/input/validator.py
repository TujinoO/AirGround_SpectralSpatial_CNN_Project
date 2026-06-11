"""
Input validation module.
输入验证模块。
"""

from typing import Dict
import numpy as np
from ..config import ProcessingConfig


class InputValidator:
    """
    Validates all input data before processing.
    在处理前验证所有输入数据。
    """
    
    @staticmethod
    def validate_image(image: np.ndarray) -> None:
        """
        Validate hyperspectral image.
        验证高光谱影像。
        
        Raises:
            ValueError: If image is invalid (如果影像无效)
        """
        if image.ndim != 3:
            raise ValueError(f"Image must be 3D, got {image.ndim}D")
        
        H, W, bands = image.shape
        if bands != 297:
            raise ValueError(f"Expected 297 bands, got {bands}")
        
        if H <= 0 or W <= 0:
            raise ValueError(f"Invalid image dimensions: ({H}, {W})")
        
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            raise ValueError("Image contains NaN or Inf values")
    
    @staticmethod
    def validate_reference_spectra(spectra: Dict[int, np.ndarray]) -> None:
        """
        Validate reference spectra.
        验证参考光谱。
        
        Raises:
            ValueError: If spectra are invalid (如果光谱无效)
        """
        if len(spectra) != 5:
            raise ValueError(f"Expected 5 classes, got {len(spectra)}")
        
        for class_id, spectrum in spectra.items():
            if spectrum.shape[0] != 297:
                raise ValueError(f"Class {class_id}: expected 297 bands, got {spectrum.shape[0]}")
            
            if np.any(np.isnan(spectrum)) or np.any(np.isinf(spectrum)):
                raise ValueError(f"Class {class_id}: contains NaN or Inf values")
    
    @staticmethod
    def validate_wavelengths(wavelengths: np.ndarray) -> None:
        """
        Validate wavelength metadata.
        验证波长元数据。
        
        Raises:
            ValueError: If wavelengths are invalid (如果波长无效)
        """
        wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)

        if wavelengths.shape[0] != 297:
            raise ValueError(f"Expected 297 wavelengths, got {wavelengths.shape[0]}")

        if np.any(np.isnan(wavelengths)) or np.any(np.isinf(wavelengths)):
            raise ValueError("Wavelengths contains NaN or Inf values")
    
    @staticmethod
    def validate_config(config: ProcessingConfig) -> None:
        """
        Validate configuration parameters.
        验证配置参数。
        
        This method delegates to the ProcessingConfig.validate() method
        to ensure all configuration parameters are valid.
        此方法委托给ProcessingConfig.validate()方法以确保所有配置参数有效。
        
        Args:
            config: Processing configuration to validate (要验证的处理配置)
        
        Raises:
            ValueError: If any configuration parameter is invalid (如果任何配置参数无效)
            TypeError: If config is not a ProcessingConfig instance (如果config不是ProcessingConfig实例)
        """
        if not isinstance(config, ProcessingConfig):
            raise TypeError(f"Expected ProcessingConfig instance, got {type(config).__name__}")
        
        # Delegate to the config's own validation method
        # 委托给配置自己的验证方法
        config.validate()
