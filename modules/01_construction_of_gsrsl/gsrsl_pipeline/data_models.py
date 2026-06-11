"""
Data models for GSRSL pipeline.
GSRSL管道的数据模型

This module defines the core data structures used throughout the pipeline.
该模块定义了整个管道中使用的核心数据结构。
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class GroundSpectrum:
    """
    Represents a single ground spectral measurement.
    表示单个地面光谱测量
    
    Attributes:
        wavelengths: Wavelength values in nanometers, shape (N,), dtype float64
                    波长值（纳米），形状(N,)，数据类型float64
                    Valid range: [350, 2500] nm
                    有效范围：[350, 2500] nm
        
        reflectances: Reflectance values (dimensionless), shape (N,), dtype float64
                     反射率值（无量纲），形状(N,)，数据类型float64
                     Valid range: [0.0, 1.0]
                     有效范围：[0.0, 1.0]
        
        class_id: Lithology class identifier (integer 0-4)
                 岩性类别标识符（整数0-4）
                 0: Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)
                 1: Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)
                 2: Mixed-type Rich Pegmatite (混合型富矿伟晶岩)
                 3: Barren Pegmatite (贫矿伟晶岩)
                 4: Wall Rock (围岩)
        
        filename: Original filename for traceability
                 原始文件名，用于可追溯性
        
        is_anomalous: Flag indicating if any reflectance values were out of valid range
                     标志，指示是否有任何反射率值超出有效范围
                     True if any reflectance values were clipped or flagged
                     如果任何反射率值被裁剪或标记，则为True
    
    Example:
        >>> wavelengths = np.array([350.0, 351.0, 352.0], dtype=np.float64)
        >>> reflectances = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        >>> spectrum = GroundSpectrum(
        ...     wavelengths=wavelengths,
        ...     reflectances=reflectances,
        ...     class_id=0,
        ...     filename="sample_001.csv",
        ...     is_anomalous=False
        ... )
    """
    wavelengths: np.ndarray
    reflectances: np.ndarray
    class_id: int
    filename: str
    is_anomalous: bool
    
    def __post_init__(self):
        """
        Validate the data after initialization.
        初始化后验证数据
        """
        # Validate wavelengths array
        if not isinstance(self.wavelengths, np.ndarray):
            raise TypeError("wavelengths must be a numpy array")
        if self.wavelengths.dtype != np.float64:
            raise TypeError("wavelengths must have dtype float64")
        if self.wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a 1D array")
        
        # Validate reflectances array
        if not isinstance(self.reflectances, np.ndarray):
            raise TypeError("reflectances must be a numpy array")
        if self.reflectances.dtype != np.float64:
            raise TypeError("reflectances must have dtype float64")
        if self.reflectances.ndim != 1:
            raise ValueError("reflectances must be a 1D array")
        
        # Validate arrays have same length
        if len(self.wavelengths) != len(self.reflectances):
            raise ValueError(
                f"wavelengths and reflectances must have same length, "
                f"got {len(self.wavelengths)} and {len(self.reflectances)}"
            )
        
        # Validate class_id
        if not isinstance(self.class_id, int):
            raise TypeError("class_id must be an integer")
        if not (0 <= self.class_id <= 4):
            raise ValueError(
                f"class_id must be in range [0, 4], got {self.class_id}"
            )
        
        # Validate filename
        if not isinstance(self.filename, str):
            raise TypeError("filename must be a string")
        
        # Validate is_anomalous
        if not isinstance(self.is_anomalous, bool):
            raise TypeError("is_anomalous must be a boolean")
    
    def __len__(self) -> int:
        """
        Return the number of spectral data points.
        返回光谱数据点的数量
        """
        return len(self.wavelengths)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the GroundSpectrum.
        返回GroundSpectrum的字符串表示
        """
        return (
            f"GroundSpectrum(filename='{self.filename}', "
            f"class_id={self.class_id}, "
            f"n_points={len(self)}, "
            f"wavelength_range=[{self.wavelengths[0]:.1f}, {self.wavelengths[-1]:.1f}] nm, "
            f"is_anomalous={self.is_anomalous})"
        )
