"""
Unit tests for output generator module.
输出生成器模块的单元测试
"""

import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from gsrsl_pipeline.output import save_gsrsl, visualize_gsrsl


class TestSaveGSRSL:
    """Tests for save_gsrsl function"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create valid test data
        self.valid_class_means = {
            0: np.full(297, 0.3, dtype=np.float32),
            1: np.full(297, 0.4, dtype=np.float32),
            2: np.full(297, 0.5, dtype=np.float32),
            3: np.full(297, 0.6, dtype=np.float32),
            4: np.full(297, 0.7, dtype=np.float32)
        }
    
    def teardown_method(self):
        """Clean up test fixtures"""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_save_valid_gsrsl(self):
        """Test saving valid GSRSL dictionary"""
        output_path = os.path.join(self.test_dir, 'gsrsl.npy')
        
        # Save GSRSL
        save_gsrsl(self.valid_class_means, output_path)
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Load and verify content
        loaded = np.load(output_path, allow_pickle=True).item()
        assert isinstance(loaded, dict)
        assert set(loaded.keys()) == {0, 1, 2, 3, 4}
        
        for class_id in range(5):
            assert loaded[class_id].shape == (297,)
            assert loaded[class_id].dtype == np.float32
            assert np.allclose(loaded[class_id], self.valid_class_means[class_id])
    
    def test_save_with_nested_directory(self):
        """Test saving to nested directory that doesn't exist"""
        output_path = os.path.join(self.test_dir, 'nested', 'dir', 'gsrsl.npy')
        
        # Save GSRSL (should create directories)
        save_gsrsl(self.valid_class_means, output_path)
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Verify directories were created
        assert os.path.exists(os.path.dirname(output_path))
    
    def test_save_with_nan_values(self):
        """Test saving GSRSL with NaN values"""
        class_means_with_nan = {
            0: np.full(297, 0.3, dtype=np.float32),
            1: np.full(297, 0.4, dtype=np.float32),
            2: np.full(297, 0.5, dtype=np.float32),
            3: np.full(297, 0.6, dtype=np.float32),
            4: np.full(297, 0.7, dtype=np.float32)
        }
        # Set some values to NaN
        class_means_with_nan[0][0:10] = np.nan
        class_means_with_nan[2][100:150] = np.nan
        
        output_path = os.path.join(self.test_dir, 'gsrsl_nan.npy')
        
        # Should save successfully
        save_gsrsl(class_means_with_nan, output_path)
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Load and verify NaN values preserved
        loaded = np.load(output_path, allow_pickle=True).item()
        assert np.sum(np.isnan(loaded[0])) == 10
        assert np.sum(np.isnan(loaded[2])) == 50
    
    def test_save_invalid_type(self):
        """Test error when class_means is not a dictionary"""
        output_path = os.path.join(self.test_dir, 'gsrsl.npy')
        
        with pytest.raises(TypeError, match="class_means must be a dictionary"):
            save_gsrsl([1, 2, 3], output_path)
    
    def test_save_missing_keys(self):
        """Test error when class_means is missing keys"""
        incomplete_means = {
            0: np.full(297, 0.3, dtype=np.float32),
            1: np.full(297, 0.4, dtype=np.float32),
            2: np.full(297, 0.5, dtype=np.float32)
            # Missing classes 3 and 4
        }
        output_path = os.path.join(self.test_dir, 'gsrsl.npy')
        
        with pytest.raises(ValueError, match="Missing keys"):
            save_gsrsl(incomplete_means, output_path)
    
    def test_save_extra_keys(self):
        """Test error when class_means has extra keys"""
        extra_means = self.valid_class_means.copy()
        extra_means[5] = np.full(297, 0.8, dtype=np.float32)
        
        output_path = os.path.join(self.test_dir, 'gsrsl.npy')
        
        with pytest.raises(ValueError, match="Extra keys"):
            save_gsrsl(extra_means, output_path)
    
    def test_save_invalid_array_type(self):
        """Test error when spectrum is not a numpy array"""
        invalid_means = self.valid_class_means.copy()
        invalid_means[0] = [0.3] * 297  # List instead of numpy array
        
        output_path = os.path.join(self.test_dir, 'gsrsl.npy')
        
        with pytest.raises(ValueError, match="spectrum must be a numpy array"):
            save_gsrsl(invalid_means, output_path)
    
    def test_save_invalid_shape(self):
        """Test error when spectrum has wrong shape"""
        invalid_means = self.valid_class_means.copy()
        invalid_means[0] = np.full(100, 0.3, dtype=np.float32)  # Wrong shape
        
        output_path = os.path.join(self.test_dir, 'gsrsl.npy')
        
        with pytest.raises(ValueError, match="spectrum must have shape \\(297,\\)"):
            save_gsrsl(invalid_means, output_path)
    
    def test_save_invalid_dtype(self):
        """Test error when spectrum has wrong dtype"""
        invalid_means = self.valid_class_means.copy()
        invalid_means[0] = np.full(297, 0.3, dtype=np.float64)  # Wrong dtype
        
        output_path = os.path.join(self.test_dir, 'gsrsl.npy')
        
        with pytest.raises(ValueError, match="spectrum must have dtype float32"):
            save_gsrsl(invalid_means, output_path)
    
    def test_save_out_of_range_values(self):
        """Test error when reflectance values are out of range"""
        invalid_means = self.valid_class_means.copy()
        invalid_means[0] = np.full(297, 1.5, dtype=np.float32)  # > 1.0
        
        output_path = os.path.join(self.test_dir, 'gsrsl.npy')
        
        with pytest.raises(ValueError, match="reflectance values must be in range"):
            save_gsrsl(invalid_means, output_path)
    
    def test_save_negative_values(self):
        """Test error when reflectance values are negative"""
        invalid_means = self.valid_class_means.copy()
        invalid_means[0] = np.full(297, -0.1, dtype=np.float32)  # < 0.0
        
        output_path = os.path.join(self.test_dir, 'gsrsl.npy')
        
        with pytest.raises(ValueError, match="reflectance values must be in range"):
            save_gsrsl(invalid_means, output_path)
    
    def test_save_boundary_values(self):
        """Test saving with boundary reflectance values (0.0 and 1.0)"""
        boundary_means = {
            0: np.full(297, 0.0, dtype=np.float32),
            1: np.full(297, 1.0, dtype=np.float32),
            2: np.full(297, 0.5, dtype=np.float32),
            3: np.full(297, 0.25, dtype=np.float32),
            4: np.full(297, 0.75, dtype=np.float32)
        }
        output_path = os.path.join(self.test_dir, 'gsrsl_boundary.npy')
        
        # Should save successfully
        save_gsrsl(boundary_means, output_path)
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Load and verify
        loaded = np.load(output_path, allow_pickle=True).item()
        assert np.allclose(loaded[0], 0.0)
        assert np.allclose(loaded[1], 1.0)
    
    def test_save_all_nan_spectrum(self):
        """Test saving when a spectrum is all NaN"""
        all_nan_means = self.valid_class_means.copy()
        all_nan_means[0] = np.full(297, np.nan, dtype=np.float32)
        
        output_path = os.path.join(self.test_dir, 'gsrsl_all_nan.npy')
        
        # Should save successfully (NaN is allowed)
        save_gsrsl(all_nan_means, output_path)
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Load and verify
        loaded = np.load(output_path, allow_pickle=True).item()
        assert np.all(np.isnan(loaded[0]))


class TestVisualizeGSRSL:
    """Tests for visualize_gsrsl function"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create valid test data
        self.valid_class_means = {
            0: np.full(297, 0.3, dtype=np.float32),
            1: np.full(297, 0.4, dtype=np.float32),
            2: np.full(297, 0.5, dtype=np.float32),
            3: np.full(297, 0.6, dtype=np.float32),
            4: np.full(297, 0.7, dtype=np.float32)
        }
        
        # Create valid wavelengths
        self.valid_wavelengths = np.linspace(400, 2500, 297, dtype=np.float64)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        # Close all matplotlib figures
        plt.close('all')
    
    def test_visualize_valid_gsrsl(self):
        """Test visualizing valid GSRSL dictionary"""
        output_path = os.path.join(self.test_dir, 'gsrsl_visualization.png')
        
        # Generate visualization
        visualize_gsrsl(self.valid_class_means, self.valid_wavelengths, output_path)
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Verify file is not empty
        assert os.path.getsize(output_path) > 0
    
    def test_visualize_with_nested_directory(self):
        """Test visualizing to nested directory that doesn't exist"""
        output_path = os.path.join(self.test_dir, 'nested', 'dir', 'visualization.png')
        
        # Generate visualization (should create directories)
        visualize_gsrsl(self.valid_class_means, self.valid_wavelengths, output_path)
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Verify directories were created
        assert os.path.exists(os.path.dirname(output_path))
    
    def test_visualize_with_nan_values(self):
        """Test visualizing GSRSL with NaN values"""
        class_means_with_nan = {
            0: np.full(297, 0.3, dtype=np.float32),
            1: np.full(297, 0.4, dtype=np.float32),
            2: np.full(297, 0.5, dtype=np.float32),
            3: np.full(297, 0.6, dtype=np.float32),
            4: np.full(297, 0.7, dtype=np.float32)
        }
        # Set some values to NaN
        class_means_with_nan[0][0:10] = np.nan
        class_means_with_nan[2][100:150] = np.nan
        
        output_path = os.path.join(self.test_dir, 'visualization_nan.png')
        
        # Should visualize successfully (matplotlib handles NaN as gaps)
        visualize_gsrsl(class_means_with_nan, self.valid_wavelengths, output_path)
        
        # Verify file exists
        assert os.path.exists(output_path)
    
    def test_visualize_invalid_class_means_type(self):
        """Test error when class_means is not a dictionary"""
        output_path = os.path.join(self.test_dir, 'visualization.png')
        
        with pytest.raises(TypeError, match="class_means must be a dictionary"):
            visualize_gsrsl([1, 2, 3], self.valid_wavelengths, output_path)
    
    def test_visualize_invalid_wavelengths_type(self):
        """Test error when wavelengths is not a numpy array"""
        output_path = os.path.join(self.test_dir, 'visualization.png')
        
        with pytest.raises(TypeError, match="wavelengths must be a numpy array"):
            visualize_gsrsl(self.valid_class_means, [400, 500, 600], output_path)
    
    def test_visualize_missing_keys(self):
        """Test error when class_means is missing keys"""
        incomplete_means = {
            0: np.full(297, 0.3, dtype=np.float32),
            1: np.full(297, 0.4, dtype=np.float32),
            2: np.full(297, 0.5, dtype=np.float32)
            # Missing classes 3 and 4
        }
        output_path = os.path.join(self.test_dir, 'visualization.png')
        
        with pytest.raises(ValueError, match="Missing keys"):
            visualize_gsrsl(incomplete_means, self.valid_wavelengths, output_path)
    
    def test_visualize_extra_keys(self):
        """Test error when class_means has extra keys"""
        extra_means = self.valid_class_means.copy()
        extra_means[5] = np.full(297, 0.8, dtype=np.float32)
        
        output_path = os.path.join(self.test_dir, 'visualization.png')
        
        with pytest.raises(ValueError, match="Extra keys"):
            visualize_gsrsl(extra_means, self.valid_wavelengths, output_path)
    
    def test_visualize_invalid_wavelengths_shape(self):
        """Test error when wavelengths has wrong shape"""
        invalid_wavelengths = np.linspace(400, 2500, 100)  # Wrong shape
        output_path = os.path.join(self.test_dir, 'visualization.png')
        
        with pytest.raises(ValueError, match="wavelengths must have shape \\(297,\\)"):
            visualize_gsrsl(self.valid_class_means, invalid_wavelengths, output_path)
    
    def test_visualize_invalid_spectrum_type(self):
        """Test error when spectrum is not a numpy array"""
        invalid_means = self.valid_class_means.copy()
        invalid_means[0] = [0.3] * 297  # List instead of numpy array
        
        output_path = os.path.join(self.test_dir, 'visualization.png')
        
        with pytest.raises(ValueError, match="spectrum must be a numpy array"):
            visualize_gsrsl(invalid_means, self.valid_wavelengths, output_path)
    
    def test_visualize_invalid_spectrum_shape(self):
        """Test error when spectrum has wrong shape"""
        invalid_means = self.valid_class_means.copy()
        invalid_means[0] = np.full(100, 0.3, dtype=np.float32)  # Wrong shape
        
        output_path = os.path.join(self.test_dir, 'visualization.png')
        
        with pytest.raises(ValueError, match="spectrum must have shape \\(297,\\)"):
            visualize_gsrsl(invalid_means, self.valid_wavelengths, output_path)
    
    def test_visualize_realistic_spectra(self):
        """Test visualizing with realistic spectral shapes"""
        # Create more realistic spectra with absorption features
        wavelengths = np.linspace(400, 2500, 297, dtype=np.float64)
        
        realistic_means = {}
        for class_id in range(5):
            # Create base spectrum with some variation
            base_level = 0.3 + class_id * 0.1
            spectrum = np.full(297, base_level, dtype=np.float32)
            
            # Add some absorption features (dips)
            for i in range(0, 297, 50):
                if i + 10 < 297:
                    spectrum[i:i+10] *= 0.8
            
            realistic_means[class_id] = spectrum
        
        output_path = os.path.join(self.test_dir, 'visualization_realistic.png')
        
        # Should visualize successfully
        visualize_gsrsl(realistic_means, wavelengths, output_path)
        
        # Verify file exists and has reasonable size
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 10000  # At least 10KB for a plot


class TestIntegration:
    """Integration tests for save and visualize together"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
        self.valid_class_means = {
            0: np.full(297, 0.3, dtype=np.float32),
            1: np.full(297, 0.4, dtype=np.float32),
            2: np.full(297, 0.5, dtype=np.float32),
            3: np.full(297, 0.6, dtype=np.float32),
            4: np.full(297, 0.7, dtype=np.float32)
        }
        
        self.valid_wavelengths = np.linspace(400, 2500, 297, dtype=np.float64)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        plt.close('all')
    
    def test_save_and_visualize_workflow(self):
        """Test complete workflow: save GSRSL and generate visualization"""
        gsrsl_path = os.path.join(self.test_dir, 'gsrsl.npy')
        viz_path = os.path.join(self.test_dir, 'gsrsl_visualization.png')
        
        # Save GSRSL
        save_gsrsl(self.valid_class_means, gsrsl_path)
        assert os.path.exists(gsrsl_path)
        
        # Generate visualization
        visualize_gsrsl(self.valid_class_means, self.valid_wavelengths, viz_path)
        assert os.path.exists(viz_path)
        
        # Load saved GSRSL and verify
        loaded = np.load(gsrsl_path, allow_pickle=True).item()
        assert set(loaded.keys()) == {0, 1, 2, 3, 4}
        
        # Verify both files have reasonable sizes
        assert os.path.getsize(gsrsl_path) > 0
        assert os.path.getsize(viz_path) > 10000
    
    def test_round_trip_with_visualization(self):
        """Test saving, loading, and visualizing GSRSL"""
        gsrsl_path = os.path.join(self.test_dir, 'gsrsl.npy')
        viz_path = os.path.join(self.test_dir, 'gsrsl_visualization.png')
        
        # Save GSRSL
        save_gsrsl(self.valid_class_means, gsrsl_path)
        
        # Load GSRSL
        loaded_means = np.load(gsrsl_path, allow_pickle=True).item()
        
        # Visualize loaded GSRSL
        visualize_gsrsl(loaded_means, self.valid_wavelengths, viz_path)
        
        # Verify visualization was created
        assert os.path.exists(viz_path)
        
        # Verify loaded data matches original
        for class_id in range(5):
            assert np.allclose(
                loaded_means[class_id],
                self.valid_class_means[class_id]
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
