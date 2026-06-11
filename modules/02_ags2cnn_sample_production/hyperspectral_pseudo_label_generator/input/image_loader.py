"""
Hyperspectral image loader module.
高光谱影像加载器模块。
"""

import numpy as np
import logging


logger = logging.getLogger(__name__)


class HyperspectralImageLoader:
    """
    Loads hyperspectral images from GeoTIFF or .npy format.
    从GeoTIFF或.npy格式加载高光谱影像。
    """
    
    @staticmethod
    def load(file_path: str) -> np.ndarray:
        """
        Load hyperspectral image.
        加载高光谱影像。
        
        Args:
            file_path: Path to image file (影像文件路径)
            
        Returns:
            np.ndarray: Image array with shape (H, W, 297)
                       形状为(H, W, 297)的影像数组
                       
        Raises:
            ValueError: If file format is unsupported or data is invalid
                       如果文件格式不支持或数据无效
        """
        if file_path.endswith('.tif') or file_path.endswith('.tiff'):
            return HyperspectralImageLoader._load_geotiff(file_path)
        elif file_path.endswith('.npy'):
            return HyperspectralImageLoader._load_npy(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    @staticmethod
    def _load_geotiff(file_path: str) -> np.ndarray:
        """Load from GeoTIFF using rasterio (使用rasterio从GeoTIFF加载)"""
        import rasterio
        with rasterio.open(file_path) as src:
            try:
                image = src.read()
                return np.transpose(image, (1, 2, 0))
            except Exception:
                logger.warning("GeoTIFF bulk read failed, switching to band-wise fallback: %s", file_path)

            band_count = src.count
            height = src.height
            width = src.width
            dtype = np.dtype(src.dtypes[0])
            image = np.zeros((band_count, height, width), dtype=dtype)
            valid = np.zeros(band_count, dtype=bool)

            for idx in range(1, band_count + 1):
                try:
                    image[idx - 1] = src.read(idx)
                    valid[idx - 1] = True
                except Exception as exc:
                    logger.warning("Failed to read band %d from %s: %s", idx, file_path, exc)

            if not np.any(valid):
                raise IOError(f"Failed to read all bands from GeoTIFF: {file_path}")

            if not np.all(valid):
                image = HyperspectralImageLoader._repair_missing_bands(image, valid)

            return np.transpose(image, (1, 2, 0))

    @staticmethod
    def _repair_missing_bands(cube: np.ndarray, valid: np.ndarray) -> np.ndarray:
        repaired = cube.copy()
        valid_indices = np.where(valid)[0]

        for idx in np.where(~valid)[0]:
            left_candidates = valid_indices[valid_indices < idx]
            right_candidates = valid_indices[valid_indices > idx]

            left_idx = int(left_candidates[-1]) if left_candidates.size > 0 else None
            right_idx = int(right_candidates[0]) if right_candidates.size > 0 else None

            if left_idx is not None and right_idx is not None:
                repaired[idx] = ((repaired[left_idx].astype(np.float64) + repaired[right_idx].astype(np.float64)) / 2.0).astype(cube.dtype)
            elif left_idx is not None:
                repaired[idx] = repaired[left_idx]
            elif right_idx is not None:
                repaired[idx] = repaired[right_idx]
            else:
                repaired[idx] = 0

        return repaired
    
    @staticmethod
    def _load_npy(file_path: str) -> np.ndarray:
        """Load from .npy format (从.npy格式加载)"""
        return np.load(file_path)
