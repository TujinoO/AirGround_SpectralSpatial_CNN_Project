"""
Unit tests for spectrum loader module
光谱加载器模块的单元测试
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from gsrsl_pipeline.spectrum_loader import load_ground_spectrum
from gsrsl_pipeline.data_models import GroundSpectrum


class TestSpectrumLoader:
    """Test suite for load_ground_spectrum function"""
    
    def create_test_spectrum_file(self, wavelengths, reflectances, delimiter=','):
        """Helper function to create a temporary spectrum file"""
        content = []
        for w, r in zip(wavelengths, reflectances):
            content.append(f"{w}{delimiter}{r}")
        
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.csv'
        ) as f:
            f.write('\n'.join(content))
            return f.name
    
    def create_label_table(self, filenames, class_ids):
        """Helper function to create a label table DataFrame"""
        return pd.DataFrame({
            'filename': filenames,
            'class_id': class_ids
        })
    
    def test_load_valid_spectrum_csv(self):
        """Test loading a valid TSG8 CSV file"""
        wavelengths = [350.0, 400.0, 450.0, 500.0]
        reflectances = [0.1, 0.2, 0.3, 0.4]
        
        filepath = self.create_test_spectrum_file(wavelengths, reflectances)
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [0])
        
        try:
            spectrum = load_ground_spectrum(filepath, label_table)
            
            assert isinstance(spectrum, GroundSpectrum)
            assert len(spectrum) == 4
            assert np.allclose(spectrum.wavelengths, wavelengths)
            assert np.allclose(spectrum.reflectances, reflectances)
            assert spectrum.class_id == 0
            assert spectrum.filename == filename
            assert spectrum.is_anomalous is False
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_with_tab_delimiter(self):
        """Test loading spectrum file with tab delimiter"""
        wavelengths = [350.0, 400.0, 450.0]
        reflectances = [0.1, 0.2, 0.3]
        
        filepath = self.create_test_spectrum_file(
            wavelengths, reflectances, delimiter='\t'
        )
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [1])
        
        try:
            spectrum = load_ground_spectrum(filepath, label_table)
            
            assert len(spectrum) == 3
            assert spectrum.class_id == 1
            assert spectrum.is_anomalous is False
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_with_space_delimiter(self):
        """Test loading spectrum file with space delimiter"""
        wavelengths = [350.0, 400.0]
        reflectances = [0.1, 0.2]
        
        filepath = self.create_test_spectrum_file(
            wavelengths, reflectances, delimiter=' '
        )
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [2])
        
        try:
            spectrum = load_ground_spectrum(filepath, label_table)
            
            assert len(spectrum) == 2
            assert spectrum.class_id == 2
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_all_class_ids(self):
        """Test loading spectra with all valid class IDs (0-4)"""
        wavelengths = [350.0, 400.0]
        reflectances = [0.1, 0.2]
        
        for class_id in range(5):
            filepath = self.create_test_spectrum_file(wavelengths, reflectances)
            filename = os.path.basename(filepath)
            label_table = self.create_label_table([filename], [class_id])
            
            try:
                spectrum = load_ground_spectrum(filepath, label_table)
                assert spectrum.class_id == class_id
            finally:
                os.unlink(filepath)
    
    def test_load_spectrum_wavelength_at_lower_boundary(self):
        """Test edge case: wavelength at lower boundary (350 nm)"""
        wavelengths = [350.0, 400.0, 450.0]
        reflectances = [0.1, 0.2, 0.3]
        
        filepath = self.create_test_spectrum_file(wavelengths, reflectances)
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [0])
        
        try:
            spectrum = load_ground_spectrum(filepath, label_table)
            assert spectrum.wavelengths[0] == 350.0
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_wavelength_at_upper_boundary(self):
        """Test edge case: wavelength at upper boundary (2500 nm)"""
        wavelengths = [2400.0, 2450.0, 2500.0]
        reflectances = [0.1, 0.2, 0.3]
        
        filepath = self.create_test_spectrum_file(wavelengths, reflectances)
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [0])
        
        try:
            spectrum = load_ground_spectrum(filepath, label_table)
            assert spectrum.wavelengths[-1] == 2500.0
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_reflectance_at_boundaries(self):
        """Test edge case: reflectances at boundaries (0.0 and 1.0)"""
        wavelengths = [350.0, 400.0, 450.0]
        reflectances = [0.0, 0.5, 1.0]
        
        filepath = self.create_test_spectrum_file(wavelengths, reflectances)
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [0])
        
        try:
            spectrum = load_ground_spectrum(filepath, label_table)
            assert spectrum.reflectances[0] == 0.0
            assert spectrum.reflectances[-1] == 1.0
            assert spectrum.is_anomalous is False
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_reflectance_below_range(self):
        """Test anomalous reflectance: values below 0.0"""
        wavelengths = [350.0, 400.0, 450.0]
        reflectances = [-0.1, 0.2, 0.3]  # One value below 0.0
        
        filepath = self.create_test_spectrum_file(wavelengths, reflectances)
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [0])
        
        try:
            spectrum = load_ground_spectrum(filepath, label_table)
            
            # Should be flagged as anomalous
            assert spectrum.is_anomalous is True
            # Value should be clipped to 0.0
            assert spectrum.reflectances[0] == 0.0
            assert spectrum.reflectances[1] == 0.2
            assert spectrum.reflectances[2] == 0.3
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_reflectance_above_range(self):
        """Test anomalous reflectance: values above 1.0"""
        wavelengths = [350.0, 400.0, 450.0]
        reflectances = [0.1, 1.5, 0.3]  # One value above 1.0
        
        filepath = self.create_test_spectrum_file(wavelengths, reflectances)
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [0])
        
        try:
            spectrum = load_ground_spectrum(filepath, label_table)
            
            # Should be flagged as anomalous
            assert spectrum.is_anomalous is True
            # Value should be clipped to 1.0
            assert spectrum.reflectances[0] == 0.1
            assert spectrum.reflectances[1] == 1.0
            assert spectrum.reflectances[2] == 0.3
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_multiple_anomalous_reflectances(self):
        """Test multiple anomalous reflectance values"""
        wavelengths = [350.0, 400.0, 450.0, 500.0]
        reflectances = [-0.2, 0.5, 1.3, -0.1]  # Multiple out-of-range values
        
        filepath = self.create_test_spectrum_file(wavelengths, reflectances)
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [0])
        
        try:
            spectrum = load_ground_spectrum(filepath, label_table)
            
            assert spectrum.is_anomalous is True
            # All values should be clipped to [0.0, 1.0]
            assert np.all(spectrum.reflectances >= 0.0)
            assert np.all(spectrum.reflectances <= 1.0)
            assert spectrum.reflectances[0] == 0.0  # Clipped from -0.2
            assert spectrum.reflectances[1] == 0.5  # Unchanged
            assert spectrum.reflectances[2] == 1.0  # Clipped from 1.3
            assert spectrum.reflectances[3] == 0.0  # Clipped from -0.1
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_file_not_found(self):
        """Test error case: spectrum file doesn't exist"""
        label_table = self.create_label_table(['nonexistent.csv'], [0])
        
        with pytest.raises(FileNotFoundError, match="Spectrum file not found"):
            load_ground_spectrum('/nonexistent/path/to/spectrum.csv', label_table)
    
    def test_load_spectrum_filename_not_in_label_table(self):
        """Test error case: filename not in label table"""
        wavelengths = [350.0, 400.0]
        reflectances = [0.1, 0.2]
        
        filepath = self.create_test_spectrum_file(wavelengths, reflectances)
        filename = os.path.basename(filepath)
        # Create label table without this filename
        label_table = self.create_label_table(['other_file.csv'], [0])
        
        try:
            with pytest.raises(ValueError, match="not found in label table"):
                load_ground_spectrum(filepath, label_table)
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_invalid_label_table_columns(self):
        """Test error case: label table missing required columns"""
        wavelengths = [350.0, 400.0]
        reflectances = [0.1, 0.2]
        
        filepath = self.create_test_spectrum_file(wavelengths, reflectances)
        # Create label table with wrong columns
        label_table = pd.DataFrame({'wrong_column': ['test.csv']})
        
        try:
            with pytest.raises(ValueError, match="must contain 'filename' and 'class_id' columns"):
                load_ground_spectrum(filepath, label_table)
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_invalid_class_id_in_table(self):
        """Test error case: invalid class_id in label table"""
        wavelengths = [350.0, 400.0]
        reflectances = [0.1, 0.2]
        
        filepath = self.create_test_spectrum_file(wavelengths, reflectances)
        filename = os.path.basename(filepath)
        # Create label table with invalid class_id
        label_table = self.create_label_table([filename], [5])  # Invalid: > 4
        
        try:
            with pytest.raises(ValueError, match="Invalid class_id"):
                load_ground_spectrum(filepath, label_table)
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_wavelength_below_range(self):
        """Test error case: wavelength below 350 nm"""
        wavelengths = [300.0, 400.0, 450.0]  # 300 < 350
        reflectances = [0.1, 0.2, 0.3]
        
        filepath = self.create_test_spectrum_file(wavelengths, reflectances)
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [0])
        
        try:
            with pytest.raises(ValueError, match="must be in range \\[350, 2500\\] nm"):
                load_ground_spectrum(filepath, label_table)
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_wavelength_above_range(self):
        """Test error case: wavelength above 2500 nm"""
        wavelengths = [2400.0, 2500.0, 2600.0]  # 2600 > 2500
        reflectances = [0.1, 0.2, 0.3]
        
        filepath = self.create_test_spectrum_file(wavelengths, reflectances)
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [0])
        
        try:
            with pytest.raises(ValueError, match="must be in range \\[350, 2500\\] nm"):
                load_ground_spectrum(filepath, label_table)
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_missing_columns(self):
        """Test error case: file with only one column"""
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.csv'
        ) as f:
            f.write("350.0\n400.0\n450.0\n")  # Only one column
            filepath = f.name
        
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [0])
        
        try:
            with pytest.raises(ValueError, match="Expected CSV/TXT format with two columns"):
                load_ground_spectrum(filepath, label_table)
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_non_numeric_data(self):
        """Test error case: non-numeric data in columns"""
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.csv'
        ) as f:
            f.write("350.0,0.1\n400.0,invalid\n450.0,0.3\n")
            filepath = f.name
        
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [0])
        
        try:
            with pytest.raises(ValueError, match="Failed to convert data to numeric values"):
                load_ground_spectrum(filepath, label_table)
        finally:
            os.unlink(filepath)
    
    def test_load_spectrum_empty_file(self):
        """Test error case: empty file"""
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.csv'
        ) as f:
            f.write("")  # Empty file
            filepath = f.name
        
        filename = os.path.basename(filepath)
        label_table = self.create_label_table([filename], [0])
        
        try:
            with pytest.raises(ValueError):
                load_ground_spectrum(filepath, label_table)
        finally:
            os.unlink(filepath)
