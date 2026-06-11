"""
Metadata parser module for extracting wavelength information.
用于提取波长信息的元数据解析器模块。
"""

import numpy as np
from typing import Tuple
import re


class MetadataParser:
    """
    Parses satellite metadata to extract wavelength information.
    解析卫星元数据以提取波长信息。
    """
    
    @staticmethod
    def parse_wavelengths(metadata_path: str) -> np.ndarray:
        """
        Extract center wavelengths from metadata.
        从元数据中提取中心波长。
        
        Args:
            metadata_path: Path to metadata file (JSON or CSV)
                          元数据文件路径 (JSON或CSV)
            
        Returns:
            np.ndarray: Wavelengths in nm, shape (297,)
                       波长(纳米)，形状 (297,)
                       
        Raises:
            ValueError: If wavelengths are invalid
                       如果波长无效
        """
        if metadata_path.endswith('.json'):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                wavelengths = np.asarray(metadata['wavelengths'], dtype=float).reshape(-1)
        elif metadata_path.endswith('.csv'):
            wavelengths = np.asarray(np.loadtxt(metadata_path, delimiter=','), dtype=float).reshape(-1)
        elif metadata_path.endswith('.txt'):
            with open(metadata_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.read().splitlines()
            values = []
            pattern = re.compile(r'^\s*Wavelengths\s+\d+\s*=\s*([0-9.]+)\s*$')
            for line in lines:
                match = pattern.match(line)
                if match:
                    values.append(float(match.group(1)))
            wavelengths = np.asarray(values, dtype=float).reshape(-1)
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path}")
        
        return wavelengths
    
    @staticmethod
    def identify_aloh_band(wavelengths: np.ndarray, 
                          min_wavelength: float = 2150.0, 
                          max_wavelength: float = 2250.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify the Al-OH absorption band region from wavelength data.
        从波长数据中识别Al-OH吸收波段区域。
        
        The Al-OH absorption band is a critical spectral region around 2200nm
        used for mineral identification in hyperspectral analysis.
        Al-OH吸收波段是2200nm附近用于高光谱分析中矿物识别的关键光谱区域。
        
        Args:
            wavelengths: Array of center wavelengths in nm, shape (297,)
                        中心波长数组(纳米)，形状 (297,)
            min_wavelength: Minimum wavelength of Al-OH band (default 2150nm)
                           Al-OH波段的最小波长(默认2150nm)
            max_wavelength: Maximum wavelength of Al-OH band (default 2250nm)
                           Al-OH波段的最大波长(默认2250nm)
            
        Returns:
            Tuple containing:
            包含以下内容的元组：
            - mask: Boolean array indicating Al-OH band indices, shape (297,)
                   指示Al-OH波段索引的布尔数组，形状 (297,)
            - indices: Integer array of band indices in Al-OH region
                      Al-OH区域中的波段索引整数数组
                      
        Raises:
            ValueError: If no wavelengths found in Al-OH region
                       如果在Al-OH区域未找到波长
        """
        # Create boolean mask for wavelengths in Al-OH range
        # 为Al-OH范围内的波长创建布尔掩码
        mask = (wavelengths >= min_wavelength) & (wavelengths <= max_wavelength)
        
        # Get indices of bands in Al-OH region
        # 获取Al-OH区域中的波段索引
        indices = np.where(mask)[0]
        
        # Validate that at least one band is in the Al-OH region
        # 验证至少有一个波段在Al-OH区域
        if len(indices) == 0:
            raise ValueError(
                f"No wavelengths found in Al-OH absorption region "
                f"({min_wavelength}-{max_wavelength} nm). "
                f"Wavelength range: {wavelengths.min():.1f}-{wavelengths.max():.1f} nm"
            )
        
        return mask, indices
