"""
Ground spectrum loader module for GSRSL pipeline.
GSRSL管道的地面光谱加载器模块

This module provides functionality to load and validate ground spectral measurements
from TSG8 software output files (CSV/TXT format).
该模块提供从TSG8软件输出文件（CSV/TXT格式）加载和验证地面光谱测量的功能。
"""

import os
import logging
from typing import Optional, List
import numpy as np
import pandas as pd

from .data_models import GroundSpectrum

# Configure logger
logger = logging.getLogger(__name__)


def load_ground_spectrum(
    filepath: str,
    label_table: pd.DataFrame
) -> GroundSpectrum:
    """
    Load a single ground spectrum file and associate with lithology class.
    加载单个地面光谱文件并与岩性类别关联
    
    This function reads TSG8 spectral data files (CSV or TXT format) and performs
    comprehensive validation on wavelength and reflectance values. It maps the
    spectrum to its lithology class using the provided label table.
    
    该函数读取TSG8光谱数据文件（CSV或TXT格式），并对波长和反射率值进行全面验证。
    它使用提供的标签表将光谱映射到其岩性类别。
    
    Args:
        filepath: Path to TSG8 CSV/TXT file containing spectral data
                 包含光谱数据的TSG8 CSV/TXT文件的路径
                 Expected format: Two columns (wavelength, reflectance)
                 预期格式：两列（波长，反射率）
        
        label_table: DataFrame with columns ['filename', 'class_id']
                    包含列['filename', 'class_id']的DataFrame
                    Maps filenames to lithology class IDs (0-4)
                    将文件名映射到岩性类别ID（0-4）
    
    Returns:
        GroundSpectrum object with loaded and validated data
        包含加载和验证数据的GroundSpectrum对象
    
    Raises:
        FileNotFoundError: If spectrum file doesn't exist
                          如果光谱文件不存在
        
        ValueError: If file format is invalid, filename not in label_table,
                   or data validation fails
                   如果文件格式无效、文件名不在label_table中或数据验证失败
    
    Example:
        >>> label_table = pd.DataFrame({
        ...     'filename': ['sample_001.csv', 'sample_002.csv'],
        ...     'class_id': [0, 1]
        ... })
        >>> spectrum = load_ground_spectrum('data/sample_001.csv', label_table)
        >>> print(spectrum)
        GroundSpectrum(filename='sample_001.csv', class_id=0, ...)
    
    Validation Rules:
    验证规则：
        - Wavelengths must be in range [350, 2500] nm
          波长必须在[350, 2500] nm范围内
        - Reflectances must be in range [0.0, 1.0]
          反射率必须在[0.0, 1.0]范围内
        - Out-of-range reflectances trigger anomaly flag and warning
          超出范围的反射率触发异常标志和警告
        - Class ID must be in range [0, 4]
          类别ID必须在[0, 4]范围内
    """
    # Validate file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Spectrum file not found: '{filepath}'. "
            f"Please verify the file path is correct."
        )
    
    # Extract filename from path
    filename = os.path.basename(filepath)
    
    # Look up class_id from label table
    if 'filename' not in label_table.columns or 'class_id' not in label_table.columns:
        raise ValueError(
            f"Label table must contain 'filename' and 'class_id' columns. "
            f"Found columns: {list(label_table.columns)}"
        )
    
    matching_rows = label_table[label_table['filename'] == filename]
    
    if len(matching_rows) == 0:
        raise ValueError(
            f"Filename '{filename}' not found in label table. "
            f"Please ensure the label table contains an entry for this file."
        )
    
    class_id = int(matching_rows.iloc[0]['class_id'])
    
    # Validate class_id range
    if not (0 <= class_id <= 4):
        raise ValueError(
            f"Invalid class_id {class_id} for file '{filename}'. "
            f"Class ID must be in range [0, 4]. "
            f"Valid classes: 0=Spodumene-rich, 1=Lepidolite-rich, "
            f"2=Mixed-type, 3=Barren, 4=Wall Rock"
        )
    
    # Load spectral data with flexible delimiter detection
    try:
        # Try common delimiters: comma, tab, space
        data = None
        for delimiter in [',', '\t', r'\s+']:
            try:
                data = pd.read_csv(
                    filepath,
                    delimiter=delimiter,
                    header=None,
                    engine='python' if delimiter == r'\s+' else 'c'
                )
                # Check if we got at least 2 columns
                if data.shape[1] >= 2:
                    break
            except Exception:
                continue
        
        if data is None or data.shape[1] < 2:
            raise ValueError("Could not parse file with any delimiter")
        
    except Exception as e:
        raise ValueError(
            f"Failed to parse spectrum file '{filepath}'. "
            f"Expected CSV/TXT format with two columns (wavelength, reflectance). "
            f"Error: {str(e)}"
        )
    
    # Validate we have at least 2 columns
    if data.shape[1] < 2:
        raise ValueError(
            f"Spectrum file '{filepath}' must have at least 2 columns. "
            f"Found {data.shape[1]} column(s). "
            f"Expected format: column 0 = wavelength (nm), column 1 = reflectance"
        )
    
    # Extract wavelengths (column 0) and reflectances (column 1)
    try:
        wavelengths = data.iloc[:, 0].values.astype(np.float64)
        reflectances = data.iloc[:, 1].values.astype(np.float64)
    except Exception as e:
        raise ValueError(
            f"Failed to convert data to numeric values in file '{filepath}'. "
            f"Ensure columns 0 and 1 contain numeric data. "
            f"Error: {str(e)}"
        )
    
    # Validate wavelength range [350, 2500] nm
    if np.any(wavelengths < 350.0) or np.any(wavelengths > 2500.0):
        invalid_wavelengths = wavelengths[(wavelengths < 350.0) | (wavelengths > 2500.0)]
        raise ValueError(
            f"Wavelengths in file '{filepath}' must be in range [350, 2500] nm. "
            f"Found {len(invalid_wavelengths)} invalid wavelength(s). "
            f"Range of invalid values: [{invalid_wavelengths.min():.2f}, {invalid_wavelengths.max():.2f}] nm"
        )
    
    # Validate and handle reflectance range [0.0, 1.0]
    is_anomalous = False
    
    if np.any(reflectances < 0.0) or np.any(reflectances > 1.0):
        is_anomalous = True
        
        # Count and report out-of-range values
        n_below = np.sum(reflectances < 0.0)
        n_above = np.sum(reflectances > 1.0)
        
        logger.warning(
            f"Anomalous reflectance values detected in file '{filename}': "
            f"{n_below} value(s) below 0.0, {n_above} value(s) above 1.0. "
            f"Values will be clipped to valid range [0.0, 1.0]. "
            f"Spectrum flagged as anomalous."
        )
        
        # Clip values to valid range
        reflectances = np.clip(reflectances, 0.0, 1.0)
    
    # Create and return GroundSpectrum object
    spectrum = GroundSpectrum(
        wavelengths=wavelengths,
        reflectances=reflectances,
        class_id=class_id,
        filename=filename,
        is_anomalous=is_anomalous
    )
    
    logger.info(
        f"Successfully loaded spectrum '{filename}': "
        f"class_id={class_id}, n_points={len(spectrum)}, "
        f"wavelength_range=[{wavelengths[0]:.1f}, {wavelengths[-1]:.1f}] nm, "
        f"anomalous={is_anomalous}"
    )
    
    return spectrum


def load_spectra_from_matrix(
    filepath: str,
    label_table: pd.DataFrame
) -> List[GroundSpectrum]:
    """
    Load multiple ground spectra from a single matrix-format CSV/TXT file.
    从单个矩阵格式的CSV/TXT文件中加载多个地面光谱
    
    Expected file format:
        - First row is header
        - Column 0 header is wavelength (e.g., 'Wavelength_(nm)')
        - Columns 1..N headers are spectrum identifiers (e.g., '000001:hll.001')
        - Column 0 values are wavelengths in nm
        - Columns 1..N values are reflectances in range [0.0, 1.0]
    
    预期文件格式：
        - 第一行为表头
        - 第0列表头为波长（如'Wavelength_(nm)'）
        - 第1..N列表头为光谱标识符（如'000001:hll.001'）
        - 第0列数据为波长，单位nm
        - 第1..N列数据为反射率，范围[0.0, 1.0]
    
    Each spectrum column name must appear in the label table 'filename' column.
    每个光谱列名必须出现在标签表的'filename'列中。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Spectrum matrix file not found: '{filepath}'. "
            f"Please verify the file path is correct."
        )
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(
            f"Failed to read spectrum matrix file '{filepath}': {e}. "
            f"Expected CSV/TXT format with header row and columns "
            f"[wavelength, spectrum_1, spectrum_2, ...]."
        ) from e
    
    if df.shape[1] < 2:
        raise ValueError(
            f"Spectrum matrix file '{filepath}' must have at least 2 columns. "
            f"Found {df.shape[1]} column(s). "
            f"Expected format: column 0 = wavelength (nm), "
            f"columns 1..N = reflectance per spectrum."
        )
    
    try:
        wavelengths = df.iloc[:, 0].values.astype(np.float64)
    except Exception as e:
        raise ValueError(
            f"Failed to convert wavelength column to numeric values in file '{filepath}'. "
            f"Ensure the first column contains numeric wavelength data in nm. "
            f"Error: {str(e)}"
        )
    
    if np.any(wavelengths < 350.0) or np.any(wavelengths > 2500.0):
        invalid_wavelengths = wavelengths[(wavelengths < 350.0) | (wavelengths > 2500.0)]
        raise ValueError(
            f"Wavelengths in file '{filepath}' must be in range [350, 2500] nm. "
            f"Found {len(invalid_wavelengths)} invalid wavelength(s). "
            f"Range of invalid values: [{invalid_wavelengths.min():.2f}, {invalid_wavelengths.max():.2f}] nm"
        )
    
    if 'filename' not in label_table.columns or 'class_id' not in label_table.columns:
        raise ValueError(
            f"Label table must contain 'filename' and 'class_id' columns. "
            f"Found columns: {list(label_table.columns)}"
        )
    
    spectra: List[GroundSpectrum] = []
    
    for col_name in df.columns[1:]:
        filename = str(col_name)
        
        matching_rows = label_table[label_table['filename'] == filename]
        if len(matching_rows) == 0:
            raise ValueError(
                f"Spectrum column '{filename}' in file '{filepath}' not found in label table. "
                f"Please ensure the label table contains an entry for each spectrum column."
            )
        
        class_id = int(matching_rows.iloc[0]['class_id'])
        
        if not (0 <= class_id <= 4):
            raise ValueError(
                f"Invalid class_id {class_id} for spectrum '{filename}'. "
                f"Class ID must be in range [0, 4]. "
                f"Valid classes: 0=Spodumene-rich, 1=Lepidolite-rich, "
                f"2=Mixed-type, 3=Barren, 4=Wall Rock"
            )
        
        try:
            reflectances = df[col_name].values.astype(np.float64)
        except Exception as e:
            raise ValueError(
                f"Failed to convert reflectance data to numeric values for column '{filename}' "
                f"in file '{filepath}'. Ensure the column contains numeric data. "
                f"Error: {str(e)}"
            )
        
        is_anomalous = False
        
        if np.any(reflectances < 0.0) or np.any(reflectances > 1.0):
            is_anomalous = True
            
            n_below = np.sum(reflectances < 0.0)
            n_above = np.sum(reflectances > 1.0)
            
            logger.warning(
                f"Anomalous reflectance values detected in spectrum '{filename}': "
                f"{n_below} value(s) below 0.0, {n_above} value(s) above 1.0. "
                f"Values will be clipped to valid range [0.0, 1.0]. "
                f"Spectrum flagged as anomalous."
            )
            
            reflectances = np.clip(reflectances, 0.0, 1.0)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=class_id,
            filename=filename,
            is_anomalous=is_anomalous
        )
        
        logger.info(
            f"Successfully loaded spectrum '{filename}' from matrix file '{filepath}': "
            f"class_id={class_id}, n_points={len(spectrum)}, "
            f"wavelength_range=[{wavelengths[0]:.1f}, {wavelengths[-1]:.1f}] nm, "
            f"anomalous={is_anomalous}"
        )
        
        spectra.append(spectrum)
    
    logger.info(
        f"Total spectra loaded from matrix file '{filepath}': {len(spectra)}"
    )
    
    return spectra
