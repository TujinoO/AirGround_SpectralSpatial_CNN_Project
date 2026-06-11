"""
Metadata Parser Module for GSRSL Pipeline
元数据解析器模块

This module provides functionality to parse GF-5 satellite metadata files
and extract band specifications (wavelengths and FWHM values).
此模块提供解析GF-5卫星元数据文件并提取波段规格（波长和FWHM值）的功能。
"""

import re
import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)


def parse_metadata(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse GF-5 metadata file to extract wavelengths and FWHM values.
    解析GF-5元数据文件以提取波长和FWHM值
    
    Args:
        filepath: Path to GF-5 metadata text file
                 GF-5元数据文本文件的路径
    
    Returns:
        Tuple of (wavelengths, fwhms) where:
        - wavelengths: np.ndarray of shape (297,), dtype float64, center wavelengths in nm
        - fwhms: np.ndarray of shape (297,), dtype float64, FWHM values in nm
        
        返回元组(wavelengths, fwhms)，其中：
        - wavelengths: 形状为(297,)的np.ndarray，dtype为float64，中心波长(nm)
        - fwhms: 形状为(297,)的np.ndarray，dtype为float64，FWHM值(nm)
    
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If file format is invalid or contains < 297 bands
        
        引发：
        FileNotFoundError: 如果元数据文件不存在
        ValueError: 如果文件格式无效或包含<297个波段
    """
    # Check if file exists
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {filepath}")
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    except Exception as e:
        logger.error(f"Error reading metadata file {filepath}: {e}")
        raise ValueError(f"Error reading metadata file {filepath}: {e}")
    
    # Regex patterns for extracting wavelengths and FWHM values
    wavelength_pattern = r"Wavelengths\s+(\d+)\s*=\s*([\d.]+)"
    fwhm_pattern = r"FWHM\s+(\d+)\s*=\s*([\d.]+)"
    
    # Extract wavelengths
    wavelength_matches = re.findall(wavelength_pattern, content)
    fwhm_matches = re.findall(fwhm_pattern, content)
    
    # Validate number of bands found
    if len(wavelength_matches) < 297:
        logger.error(f"Insufficient wavelength bands found: {len(wavelength_matches)}/297")
        raise ValueError(
            f"Metadata file contains insufficient wavelength bands: "
            f"found {len(wavelength_matches)}, expected 297"
        )
    
    if len(fwhm_matches) < 297:
        logger.error(f"Insufficient FWHM bands found: {len(fwhm_matches)}/297")
        raise ValueError(
            f"Metadata file contains insufficient FWHM bands: "
            f"found {len(fwhm_matches)}, expected 297"
        )
    
    # Parse and sort wavelengths by band number
    try:
        wavelength_data = [(int(band_num), float(value)) for band_num, value in wavelength_matches]
        wavelength_data.sort(key=lambda x: x[0])
        wavelengths = np.array([value for _, value in wavelength_data[:297]], dtype=np.float64)
    except ValueError as e:
        logger.error(f"Error parsing wavelength values: {e}")
        raise ValueError(f"Invalid wavelength values in metadata file: {e}")
    
    # Parse and sort FWHM values by band number
    try:
        fwhm_data = [(int(band_num), float(value)) for band_num, value in fwhm_matches]
        fwhm_data.sort(key=lambda x: x[0])
        fwhms = np.array([value for _, value in fwhm_data[:297]], dtype=np.float64)
    except ValueError as e:
        logger.error(f"Error parsing FWHM values: {e}")
        raise ValueError(f"Invalid FWHM values in metadata file: {e}")
    
    # Validate array shapes
    if wavelengths.shape != (297,):
        logger.error(f"Invalid wavelength array shape: {wavelengths.shape}")
        raise ValueError(f"Invalid wavelength array shape: {wavelengths.shape}, expected (297,)")
    
    if fwhms.shape != (297,):
        logger.error(f"Invalid FWHM array shape: {fwhms.shape}")
        raise ValueError(f"Invalid FWHM array shape: {fwhms.shape}, expected (297,)")
    
    logger.info(f"Successfully parsed metadata from {filepath}: 297 bands extracted")
    
    return wavelengths, fwhms
