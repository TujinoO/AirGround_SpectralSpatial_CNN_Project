"""
Unit tests for HyperspectralImageLoader class.
HyperspectralImageLoader类的单元测试。
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from hyperspectral_pseudo_label_generator.input.image_loader import HyperspectralImageLoader


class TestHyperspectralImageLoader:
    """Test suite for HyperspectralImageLoader class."""
    
    def test_load_npy_valid_image(self):
        """Test loading a valid .npy file with correct shape."""
        # Create a temporary .npy file with valid data
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            # Create valid image data (H=10, W=10, bands=297)
            valid_image = np.random.rand(10, 10, 297).astype(np.float32)
            np.save(tmp_path, valid_image)
        
        try:
            # Load the image
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            
            # Verify shape
            assert loaded_image.shape == (10, 10, 297), \
                f"Expected shape (10, 10, 297), got {loaded_image.shape}"
            
            # Verify data is preserved
            np.testing.assert_array_almost_equal(loaded_image, valid_image)
        finally:
            # Clean up
            os.unlink(tmp_path)
    
    def test_load_npy_different_dimensions(self):
        """Test loading .npy files with different spatial dimensions."""
        test_cases = [
            (50, 50, 297),
            (100, 200, 297),
            (1, 1, 297),
            (500, 300, 297)
        ]
        
        for H, W, bands in test_cases:
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
                tmp_path = tmp.name
                valid_image = np.random.rand(H, W, bands).astype(np.float32)
                np.save(tmp_path, valid_image)
            
            try:
                loaded_image = HyperspectralImageLoader.load(tmp_path)
                assert loaded_image.shape == (H, W, bands), \
                    f"Expected shape {(H, W, bands)}, got {loaded_image.shape}"
            finally:
                os.unlink(tmp_path)
    
    def test_load_unsupported_format(self):
        """Test that unsupported file formats raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                HyperspectralImageLoader.load(tmp_path)
            
            assert "Unsupported file format" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)
    
    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises an error."""
        nonexistent_path = "/nonexistent/path/to/image.npy"
        
        with pytest.raises((FileNotFoundError, OSError)):
            HyperspectralImageLoader.load(nonexistent_path)
    
    def test_load_npy_preserves_dtype(self):
        """Test that loading preserves the data type."""
        dtypes = [np.float32, np.float64, np.int16, np.uint16]
        
        for dtype in dtypes:
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
                tmp_path = tmp.name
                if np.issubdtype(dtype, np.integer):
                    valid_image = np.random.randint(0, 1000, size=(10, 10, 297), dtype=dtype)
                else:
                    valid_image = np.random.rand(10, 10, 297).astype(dtype)
                np.save(tmp_path, valid_image)
            
            try:
                loaded_image = HyperspectralImageLoader.load(tmp_path)
                assert loaded_image.dtype == dtype, \
                    f"Expected dtype {dtype}, got {loaded_image.dtype}"
            finally:
                os.unlink(tmp_path)
    
    def test_load_npy_with_valid_values(self):
        """Test loading .npy file with various valid value ranges."""
        # Test with reflectance values [0, 1]
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            valid_image = np.random.rand(10, 10, 297).astype(np.float32)
            np.save(tmp_path, valid_image)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert np.all(loaded_image >= 0) and np.all(loaded_image <= 1)
        finally:
            os.unlink(tmp_path)
        
        # Test with DN values [0, 10000]
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            valid_image = (np.random.rand(10, 10, 297) * 10000).astype(np.float32)
            np.save(tmp_path, valid_image)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert np.all(loaded_image >= 0) and np.all(loaded_image <= 10000)
        finally:
            os.unlink(tmp_path)
    
    def test_load_geotiff_format_detection(self):
        """Test that .tif and .tiff extensions are recognized."""
        # We'll test that the method attempts to load GeoTIFF
        # This test will fail if rasterio is not installed or file doesn't exist
        # but it verifies the format detection logic
        
        # Test .tif extension
        with pytest.raises((FileNotFoundError, OSError, ImportError)):
            # This should attempt to load as GeoTIFF
            HyperspectralImageLoader.load("test_image.tif")
        
        # Test .tiff extension
        with pytest.raises((FileNotFoundError, OSError, ImportError)):
            # This should attempt to load as GeoTIFF
            HyperspectralImageLoader.load("test_image.tiff")
    
    def test_load_npy_empty_array(self):
        """Test loading an empty array raises appropriate error or handles gracefully."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            # Create an empty array
            empty_image = np.array([])
            np.save(tmp_path, empty_image)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            # The loader should return the empty array
            # Validation will happen in the InputValidator
            assert loaded_image.size == 0
        finally:
            os.unlink(tmp_path)
    
    def test_load_npy_wrong_dimensions(self):
        """Test loading arrays with wrong number of dimensions."""
        # 2D array (missing band dimension)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            wrong_image = np.random.rand(10, 10).astype(np.float32)
            np.save(tmp_path, wrong_image)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            # Loader should return the array as-is
            # Validation will happen in InputValidator
            assert loaded_image.ndim == 2
        finally:
            os.unlink(tmp_path)
        
        # 4D array (extra dimension)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            wrong_image = np.random.rand(10, 10, 297, 1).astype(np.float32)
            np.save(tmp_path, wrong_image)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            # Loader should return the array as-is
            assert loaded_image.ndim == 4
        finally:
            os.unlink(tmp_path)
    
    def test_load_npy_with_nan_values(self):
        """Test loading .npy file containing NaN values."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            image_with_nan = np.random.rand(10, 10, 297).astype(np.float32)
            image_with_nan[5, 5, 100] = np.nan
            np.save(tmp_path, image_with_nan)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            # Loader should return the array with NaN
            # Validation will happen in InputValidator
            assert np.any(np.isnan(loaded_image))
        finally:
            os.unlink(tmp_path)
    
    def test_load_npy_with_inf_values(self):
        """Test loading .npy file containing Inf values."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            image_with_inf = np.random.rand(10, 10, 297).astype(np.float32)
            image_with_inf[5, 5, 100] = np.inf
            image_with_inf[3, 3, 50] = -np.inf
            np.save(tmp_path, image_with_inf)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            # Loader should return the array with Inf
            # Validation will happen in InputValidator
            assert np.any(np.isinf(loaded_image))
        finally:
            os.unlink(tmp_path)
    
    def test_load_npy_large_image(self):
        """Test loading a larger image to verify memory handling."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            # Create a larger image (1000x1000x297 ≈ 1.1GB for float32)
            # Use a smaller size for testing to avoid memory issues
            large_image = np.random.rand(100, 100, 297).astype(np.float32)
            np.save(tmp_path, large_image)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert loaded_image.shape == (100, 100, 297)
        finally:
            os.unlink(tmp_path)
    
    def test_load_npy_with_negative_values(self):
        """Test loading .npy file with negative values (valid for some preprocessing)."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            image_with_negatives = np.random.randn(10, 10, 297).astype(np.float32)
            np.save(tmp_path, image_with_negatives)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            # Loader should handle negative values
            assert np.any(loaded_image < 0)
        finally:
            os.unlink(tmp_path)
    
    def test_load_npy_file_path_variations(self):
        """Test loading with various file path formats."""
        # Test with absolute path
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = os.path.abspath(tmp.name)
            valid_image = np.random.rand(10, 10, 297).astype(np.float32)
            np.save(tmp_path, valid_image)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert loaded_image.shape == (10, 10, 297)
        finally:
            os.unlink(tmp_path)
        
        # Test with Path object
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            valid_image = np.random.rand(10, 10, 297).astype(np.float32)
            np.save(str(tmp_path), valid_image)
        
        try:
            loaded_image = HyperspectralImageLoader.load(str(tmp_path))
            assert loaded_image.shape == (10, 10, 297)
        finally:
            os.unlink(str(tmp_path))


class TestHyperspectralImageLoaderEdgeCases:
    """Test edge cases for HyperspectralImageLoader."""
    
    def test_load_npy_single_pixel_image(self):
        """Test loading a single-pixel image (1x1x297)."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            single_pixel = np.random.rand(1, 1, 297).astype(np.float32)
            np.save(tmp_path, single_pixel)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert loaded_image.shape == (1, 1, 297)
        finally:
            os.unlink(tmp_path)
    
    def test_load_npy_single_row_image(self):
        """Test loading a single-row image (1xWx297)."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            single_row = np.random.rand(1, 100, 297).astype(np.float32)
            np.save(tmp_path, single_row)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert loaded_image.shape == (1, 100, 297)
        finally:
            os.unlink(tmp_path)
    
    def test_load_npy_single_column_image(self):
        """Test loading a single-column image (Hx1x297)."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            single_column = np.random.rand(100, 1, 297).astype(np.float32)
            np.save(tmp_path, single_column)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert loaded_image.shape == (100, 1, 297)
        finally:
            os.unlink(tmp_path)
    
    def test_load_npy_all_zeros(self):
        """Test loading an image with all zero values."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            zero_image = np.zeros((10, 10, 297), dtype=np.float32)
            np.save(tmp_path, zero_image)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert np.all(loaded_image == 0)
        finally:
            os.unlink(tmp_path)
    
    def test_load_npy_all_ones(self):
        """Test loading an image with all one values."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            ones_image = np.ones((10, 10, 297), dtype=np.float32)
            np.save(tmp_path, ones_image)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert np.all(loaded_image == 1)
        finally:
            os.unlink(tmp_path)
    
    def test_load_npy_very_large_values(self):
        """Test loading an image with very large values."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            large_values = np.random.rand(10, 10, 297).astype(np.float32) * 1e6
            np.save(tmp_path, large_values)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert np.all(loaded_image >= 0)
            assert np.max(loaded_image) > 1e5
        finally:
            os.unlink(tmp_path)

    def test_repair_missing_bands_interpolates_from_neighbors(self):
        cube = np.zeros((5, 2, 2), dtype=np.float32)
        cube[0] = 10
        cube[2] = 30
        cube[4] = 50
        valid = np.array([True, False, True, False, True])

        repaired = HyperspectralImageLoader._repair_missing_bands(cube, valid)

        np.testing.assert_array_equal(repaired[1], np.full((2, 2), 20, dtype=np.float32))
        np.testing.assert_array_equal(repaired[3], np.full((2, 2), 40, dtype=np.float32))
    
    def test_load_npy_very_small_values(self):
        """Test loading an image with very small positive values."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
            small_values = np.random.rand(10, 10, 297).astype(np.float32) * 1e-10
            np.save(tmp_path, small_values)
        
        try:
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert np.all(loaded_image >= 0)
            assert np.max(loaded_image) < 1e-8
        finally:
            os.unlink(tmp_path)
