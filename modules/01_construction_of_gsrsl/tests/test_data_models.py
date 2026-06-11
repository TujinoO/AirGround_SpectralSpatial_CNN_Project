"""
Unit tests for data models module
数据模型模块的单元测试
"""

import pytest
import numpy as np
from gsrsl_pipeline.data_models import GroundSpectrum


class TestGroundSpectrum:
    """Test suite for GroundSpectrum dataclass"""
    
    def test_create_valid_ground_spectrum(self):
        """Test creating a valid GroundSpectrum object"""
        wavelengths = np.array([350.0, 351.0, 352.0], dtype=np.float64)
        reflectances = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=0,
            filename="sample_001.csv",
            is_anomalous=False
        )
        
        assert np.array_equal(spectrum.wavelengths, wavelengths)
        assert np.array_equal(spectrum.reflectances, reflectances)
        assert spectrum.class_id == 0
        assert spectrum.filename == "sample_001.csv"
        assert spectrum.is_anomalous is False
    
    def test_ground_spectrum_length(self):
        """Test __len__ method returns correct number of points"""
        wavelengths = np.array([350.0, 351.0, 352.0, 353.0], dtype=np.float64)
        reflectances = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=1,
            filename="test.csv",
            is_anomalous=False
        )
        
        assert len(spectrum) == 4
    
    def test_ground_spectrum_repr(self):
        """Test __repr__ method returns informative string"""
        wavelengths = np.array([350.0, 400.0, 450.0], dtype=np.float64)
        reflectances = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=2,
            filename="test.csv",
            is_anomalous=True
        )
        
        repr_str = repr(spectrum)
        assert "test.csv" in repr_str
        assert "class_id=2" in repr_str
        assert "n_points=3" in repr_str
        assert "350.0" in repr_str
        assert "450.0" in repr_str
        assert "is_anomalous=True" in repr_str
    
    def test_ground_spectrum_invalid_wavelengths_type(self):
        """Test error when wavelengths is not a numpy array"""
        with pytest.raises(TypeError, match="wavelengths must be a numpy array"):
            GroundSpectrum(
                wavelengths=[350.0, 351.0],  # List instead of array
                reflectances=np.array([0.1, 0.2], dtype=np.float64),
                class_id=0,
                filename="test.csv",
                is_anomalous=False
            )
    
    def test_ground_spectrum_invalid_wavelengths_dtype(self):
        """Test error when wavelengths has wrong dtype"""
        with pytest.raises(TypeError, match="wavelengths must have dtype float64"):
            GroundSpectrum(
                wavelengths=np.array([350, 351], dtype=np.int32),
                reflectances=np.array([0.1, 0.2], dtype=np.float64),
                class_id=0,
                filename="test.csv",
                is_anomalous=False
            )
    
    def test_ground_spectrum_invalid_wavelengths_dimension(self):
        """Test error when wavelengths is not 1D"""
        with pytest.raises(ValueError, match="wavelengths must be a 1D array"):
            GroundSpectrum(
                wavelengths=np.array([[350.0, 351.0]], dtype=np.float64),
                reflectances=np.array([0.1, 0.2], dtype=np.float64),
                class_id=0,
                filename="test.csv",
                is_anomalous=False
            )
    
    def test_ground_spectrum_invalid_reflectances_type(self):
        """Test error when reflectances is not a numpy array"""
        with pytest.raises(TypeError, match="reflectances must be a numpy array"):
            GroundSpectrum(
                wavelengths=np.array([350.0, 351.0], dtype=np.float64),
                reflectances=[0.1, 0.2],  # List instead of array
                class_id=0,
                filename="test.csv",
                is_anomalous=False
            )
    
    def test_ground_spectrum_invalid_reflectances_dtype(self):
        """Test error when reflectances has wrong dtype"""
        with pytest.raises(TypeError, match="reflectances must have dtype float64"):
            GroundSpectrum(
                wavelengths=np.array([350.0, 351.0], dtype=np.float64),
                reflectances=np.array([0.1, 0.2], dtype=np.float32),
                class_id=0,
                filename="test.csv",
                is_anomalous=False
            )
    
    def test_ground_spectrum_invalid_reflectances_dimension(self):
        """Test error when reflectances is not 1D"""
        with pytest.raises(ValueError, match="reflectances must be a 1D array"):
            GroundSpectrum(
                wavelengths=np.array([350.0, 351.0], dtype=np.float64),
                reflectances=np.array([[0.1, 0.2]], dtype=np.float64),
                class_id=0,
                filename="test.csv",
                is_anomalous=False
            )
    
    def test_ground_spectrum_mismatched_lengths(self):
        """Test error when wavelengths and reflectances have different lengths"""
        with pytest.raises(ValueError, match="wavelengths and reflectances must have same length"):
            GroundSpectrum(
                wavelengths=np.array([350.0, 351.0, 352.0], dtype=np.float64),
                reflectances=np.array([0.1, 0.2], dtype=np.float64),
                class_id=0,
                filename="test.csv",
                is_anomalous=False
            )
    
    def test_ground_spectrum_invalid_class_id_type(self):
        """Test error when class_id is not an integer"""
        with pytest.raises(TypeError, match="class_id must be an integer"):
            GroundSpectrum(
                wavelengths=np.array([350.0], dtype=np.float64),
                reflectances=np.array([0.1], dtype=np.float64),
                class_id=0.5,  # Float instead of int
                filename="test.csv",
                is_anomalous=False
            )
    
    def test_ground_spectrum_invalid_class_id_range_negative(self):
        """Test error when class_id is negative"""
        with pytest.raises(ValueError, match="class_id must be in range"):
            GroundSpectrum(
                wavelengths=np.array([350.0], dtype=np.float64),
                reflectances=np.array([0.1], dtype=np.float64),
                class_id=-1,
                filename="test.csv",
                is_anomalous=False
            )
    
    def test_ground_spectrum_invalid_class_id_range_too_high(self):
        """Test error when class_id is greater than 4"""
        with pytest.raises(ValueError, match="class_id must be in range"):
            GroundSpectrum(
                wavelengths=np.array([350.0], dtype=np.float64),
                reflectances=np.array([0.1], dtype=np.float64),
                class_id=5,
                filename="test.csv",
                is_anomalous=False
            )
    
    def test_ground_spectrum_all_valid_class_ids(self):
        """Test that all valid class IDs (0-4) are accepted"""
        wavelengths = np.array([350.0], dtype=np.float64)
        reflectances = np.array([0.1], dtype=np.float64)
        
        for class_id in range(5):
            spectrum = GroundSpectrum(
                wavelengths=wavelengths,
                reflectances=reflectances,
                class_id=class_id,
                filename="test.csv",
                is_anomalous=False
            )
            assert spectrum.class_id == class_id
    
    def test_ground_spectrum_invalid_filename_type(self):
        """Test error when filename is not a string"""
        with pytest.raises(TypeError, match="filename must be a string"):
            GroundSpectrum(
                wavelengths=np.array([350.0], dtype=np.float64),
                reflectances=np.array([0.1], dtype=np.float64),
                class_id=0,
                filename=123,  # Integer instead of string
                is_anomalous=False
            )
    
    def test_ground_spectrum_invalid_is_anomalous_type(self):
        """Test error when is_anomalous is not a boolean"""
        with pytest.raises(TypeError, match="is_anomalous must be a boolean"):
            GroundSpectrum(
                wavelengths=np.array([350.0], dtype=np.float64),
                reflectances=np.array([0.1], dtype=np.float64),
                class_id=0,
                filename="test.csv",
                is_anomalous="False"  # String instead of bool
            )
