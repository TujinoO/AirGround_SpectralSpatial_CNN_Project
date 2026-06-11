"""
Unit tests for HyperspectralImageLoader GeoTIFF functionality.
HyperspectralImageLoader GeoTIFF功能的单元测试。

Note: These tests require rasterio to be installed.
注意: 这些测试需要安装rasterio。
"""

import numpy as np
import pytest
import tempfile
import os

try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from hyperspectral_pseudo_label_generator.input.image_loader import HyperspectralImageLoader


@pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not installed")
class TestHyperspectralImageLoaderGeoTIFF:
    """Test suite for GeoTIFF loading functionality."""
    
    def test_load_geotiff_valid_image(self):
        """Test loading a valid GeoTIFF file with 297 bands."""
        # Create a temporary GeoTIFF file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create test data: (bands, H, W) = (297, 10, 10)
            H, W, bands = 10, 10, 297
            test_data = np.random.rand(bands, H, W).astype(np.float32)
            
            # Write GeoTIFF
            transform = from_bounds(0, 0, W, H, W, H)
            with rasterio.open(
                tmp_path,
                'w',
                driver='GTiff',
                height=H,
                width=W,
                count=bands,
                dtype=test_data.dtype,
                transform=transform
            ) as dst:
                dst.write(test_data)
            
            # Load using our loader
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            
            # Verify shape is transposed correctly: (bands, H, W) → (H, W, bands)
            assert loaded_image.shape == (H, W, bands), \
                f"Expected shape ({H}, {W}, {bands}), got {loaded_image.shape}"
            
            # Verify data is transposed correctly
            for b in range(bands):
                np.testing.assert_array_almost_equal(
                    loaded_image[:, :, b],
                    test_data[b, :, :],
                    err_msg=f"Band {b} data mismatch after transpose"
                )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_geotiff_different_dimensions(self):
        """Test loading GeoTIFF files with different spatial dimensions."""
        test_cases = [
            (50, 50, 297),
            (100, 200, 297),
            (1, 1, 297),
        ]
        
        for H, W, bands in test_cases:
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # Create test data
                test_data = np.random.rand(bands, H, W).astype(np.float32)
                
                # Write GeoTIFF
                transform = from_bounds(0, 0, W, H, W, H)
                with rasterio.open(
                    tmp_path,
                    'w',
                    driver='GTiff',
                    height=H,
                    width=W,
                    count=bands,
                    dtype=test_data.dtype,
                    transform=transform
                ) as dst:
                    dst.write(test_data)
                
                # Load and verify
                loaded_image = HyperspectralImageLoader.load(tmp_path)
                assert loaded_image.shape == (H, W, bands), \
                    f"Expected shape ({H}, {W}, {bands}), got {loaded_image.shape}"
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    def test_load_geotiff_tiff_extension(self):
        """Test loading GeoTIFF with .tiff extension."""
        with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create test data
            H, W, bands = 10, 10, 297
            test_data = np.random.rand(bands, H, W).astype(np.float32)
            
            # Write GeoTIFF
            transform = from_bounds(0, 0, W, H, W, H)
            with rasterio.open(
                tmp_path,
                'w',
                driver='GTiff',
                height=H,
                width=W,
                count=bands,
                dtype=test_data.dtype,
                transform=transform
            ) as dst:
                dst.write(test_data)
            
            # Load and verify
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert loaded_image.shape == (H, W, bands)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_geotiff_preserves_dtype(self):
        """Test that GeoTIFF loading preserves data type."""
        dtypes = [np.float32, np.float64, np.int16, np.uint16]
        
        for dtype in dtypes:
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                H, W, bands = 10, 10, 297
                if np.issubdtype(dtype, np.integer):
                    test_data = np.random.randint(0, 1000, size=(bands, H, W), dtype=dtype)
                else:
                    test_data = np.random.rand(bands, H, W).astype(dtype)
                
                # Write GeoTIFF
                transform = from_bounds(0, 0, W, H, W, H)
                with rasterio.open(
                    tmp_path,
                    'w',
                    driver='GTiff',
                    height=H,
                    width=W,
                    count=bands,
                    dtype=test_data.dtype,
                    transform=transform
                ) as dst:
                    dst.write(test_data)
                
                # Load and verify dtype
                loaded_image = HyperspectralImageLoader.load(tmp_path)
                assert loaded_image.dtype == dtype, \
                    f"Expected dtype {dtype}, got {loaded_image.dtype}"
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    def test_load_geotiff_with_nodata(self):
        """Test loading GeoTIFF with nodata values."""
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            H, W, bands = 10, 10, 297
            test_data = np.random.rand(bands, H, W).astype(np.float32)
            # Set some nodata values
            test_data[:, 5, 5] = -9999
            
            # Write GeoTIFF with nodata value
            transform = from_bounds(0, 0, W, H, W, H)
            with rasterio.open(
                tmp_path,
                'w',
                driver='GTiff',
                height=H,
                width=W,
                count=bands,
                dtype=test_data.dtype,
                transform=transform,
                nodata=-9999
            ) as dst:
                dst.write(test_data)
            
            # Load and verify nodata is preserved
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert loaded_image.shape == (H, W, bands)
            # Check that nodata values are present
            assert np.any(loaded_image[5, 5, :] == -9999)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_geotiff_single_band(self):
        """Test loading a single-band GeoTIFF (should work but not be 297 bands)."""
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            H, W, bands = 10, 10, 1
            test_data = np.random.rand(bands, H, W).astype(np.float32)
            
            # Write GeoTIFF
            transform = from_bounds(0, 0, W, H, W, H)
            with rasterio.open(
                tmp_path,
                'w',
                driver='GTiff',
                height=H,
                width=W,
                count=bands,
                dtype=test_data.dtype,
                transform=transform
            ) as dst:
                dst.write(test_data)
            
            # Load - should work but validation will catch wrong band count
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert loaded_image.shape == (H, W, bands)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_geotiff_wrong_band_count(self):
        """Test loading GeoTIFF with incorrect number of bands."""
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            H, W, bands = 10, 10, 100  # Wrong number of bands
            test_data = np.random.rand(bands, H, W).astype(np.float32)
            
            # Write GeoTIFF
            transform = from_bounds(0, 0, W, H, W, H)
            with rasterio.open(
                tmp_path,
                'w',
                driver='GTiff',
                height=H,
                width=W,
                count=bands,
                dtype=test_data.dtype,
                transform=transform
            ) as dst:
                dst.write(test_data)
            
            # Load - should work but validation will catch wrong band count
            loaded_image = HyperspectralImageLoader.load(tmp_path)
            assert loaded_image.shape == (H, W, bands)
            # Validation will happen in InputValidator
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
