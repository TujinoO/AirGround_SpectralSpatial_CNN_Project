"""
Reference spectral library loader module.
参考光谱库加载器模块。
"""

from typing import Dict
import numpy as np
import logging
import os


class ReferenceSpectraLoader:
    """
    Loads reference spectral library.
    加载参考光谱库。
    """
    
    @staticmethod
    def load(file_path: str) -> Dict[int, np.ndarray]:
        """
        Load reference spectra for all mineral classes.
        加载所有矿物类别的参考光谱。
        
        Args:
            file_path: Path to reference spectra file (参考光谱文件路径)
            
        Returns:
            Dict mapping class index to spectrum array (5, 297)
            将类别索引映射到光谱数组的字典 (5, 297)
            Keys are class indices 0-4:
            键是类别索引0-4:
            - 0: Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)
            - 1: Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)
            - 2: Mixed-type Rich Pegmatite (混合型富矿伟晶岩)
            - 3: Barren Pegmatite (贫矿伟晶岩)
            - 4: Wall Rock (围岩)
            
        Raises:
            ValueError: If file is malformed or missing classes
                       如果文件格式错误或缺少类别
            FileNotFoundError: If file does not exist
                              如果文件不存在
            IOError: If file cannot be read
                    如果文件无法读取
        """
        logger = logging.getLogger(__name__)
        
        # Check if file exists (检查文件是否存在)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reference spectra file not found: {file_path}")
        
        logger.info(f"Loading reference spectra from: {file_path}")
        
        # Load based on file format (根据文件格式加载)
        try:
            if file_path.endswith('.npy'):
                spectra = ReferenceSpectraLoader._load_npy(file_path)
            elif file_path.endswith('.csv'):
                spectra = ReferenceSpectraLoader._load_csv(file_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_path}. "
                    f"Supported formats are .npy and .csv"
                )
        except Exception as e:
            if isinstance(e, (ValueError, FileNotFoundError)):
                raise
            raise IOError(f"Failed to load reference spectra from {file_path}: {str(e)}")
        
        # Validate shape (验证形状)
        if spectra.ndim != 2:
            raise ValueError(
                f"Reference spectra must be 2D array, got {spectra.ndim}D. "
                f"Expected shape: (5, 297)"
            )
        
        num_classes, num_bands = spectra.shape
        
        # Validate number of classes (验证类别数量)
        if num_classes != 5:
            raise ValueError(
                f"Expected 5 classes, got {num_classes}. "
                f"Reference spectra must contain exactly 5 mineral classes (0-4)."
            )
        
        # Validate number of bands (验证波段数量)
        if num_bands != 297:
            raise ValueError(
                f"Expected 297 bands, got {num_bands}. "
                f"Each spectrum must have exactly 297 spectral bands to match GF-5 imagery."
            )
        
        # Check for invalid values (检查无效值)
        if np.any(np.isnan(spectra)):
            raise ValueError(
                "Reference spectra contain NaN values. "
                "Please ensure all spectral values are valid numbers."
            )
        
        if np.any(np.isinf(spectra)):
            raise ValueError(
                "Reference spectra contain Inf values. "
                "Please ensure all spectral values are finite."
            )
        
        # Organize by class (按类别组织)
        # Classes 0-2: Ore classes (矿石类别)
        # Class 3: Barren Pegmatite (贫矿伟晶岩)
        # Class 4: Wall Rock (围岩)
        spectra_dict = {i: spectra[i] for i in range(5)}
        
        logger.info("Successfully loaded reference spectra: 5 classes, 297 bands per spectrum")
        
        return spectra_dict
    
    @staticmethod
    def _load_npy(file_path: str) -> np.ndarray:
        """
        Load reference spectra from .npy format.
        从.npy格式加载参考光谱。
        
        Args:
            file_path: Path to .npy file (.npy文件路径)
            
        Returns:
            Spectra array (光谱数组)
            
        Raises:
            IOError: If file cannot be loaded (如果文件无法加载)
        """
        try:
            try:
                loaded = np.load(file_path, allow_pickle=False)
            except ValueError as e:
                if "allow_pickle=False" not in str(e):
                    raise
                loaded = np.load(file_path, allow_pickle=True)

            if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
                loaded_obj = loaded.item()
                if isinstance(loaded_obj, dict):
                    spectra_rows = []
                    for class_id in range(5):
                        if class_id in loaded_obj:
                            spectrum = loaded_obj[class_id]
                        elif str(class_id) in loaded_obj:
                            spectrum = loaded_obj[str(class_id)]
                        else:
                            raise ValueError(f"Missing class {class_id} in reference spectra dict")
                        spectrum_arr = np.asarray(spectrum).reshape(-1)
                        spectra_rows.append(spectrum_arr)
                    return np.stack(spectra_rows, axis=0)

            return loaded
        except Exception as e:
            raise IOError(f"Failed to load .npy file: {str(e)}")
    
    @staticmethod
    def _load_csv(file_path: str) -> np.ndarray:
        """
        Load reference spectra from .csv format.
        从.csv格式加载参考光谱。
        
        Args:
            file_path: Path to .csv file (.csv文件路径)
            
        Returns:
            Spectra array (光谱数组)
            
        Raises:
            IOError: If file cannot be loaded (如果文件无法加载)
        """
        try:
            return np.loadtxt(file_path, delimiter=',')
        except Exception as e:
            raise IOError(
                f"Failed to load .csv file: {str(e)}. "
                f"Ensure the file is a valid CSV with comma-separated values."
            )
