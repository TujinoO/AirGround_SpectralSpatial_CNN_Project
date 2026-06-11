"""
Unit tests for class aggregator module.
类别聚合器模块的单元测试
"""

import pytest
import numpy as np
from gsrsl_pipeline.aggregator import aggregate_by_class


class TestAggregateByClassValidation:
    """Test input validation for aggregate_by_class."""
    
    def test_invalid_input_type(self):
        """Test that non-list input raises TypeError."""
        with pytest.raises(TypeError, match="resampled_spectra must be a list"):
            aggregate_by_class("not a list")
    
    def test_empty_list(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="resampled_spectra cannot be empty"):
            aggregate_by_class([])
    
    def test_invalid_tuple_structure(self):
        """Test that non-tuple items raise TypeError."""
        invalid_data = [
            (0, np.full(297, 0.5, dtype=np.float32)),
            "not a tuple"
        ]
        
        with pytest.raises(TypeError, match="must be a tuple"):
            aggregate_by_class(invalid_data)
    
    def test_invalid_class_id_type(self):
        """Test that non-integer class_id raises TypeError."""
        invalid_data = [
            ("0", np.full(297, 0.5, dtype=np.float32))
        ]
        
        with pytest.raises(TypeError, match="class_id must be an integer"):
            aggregate_by_class(invalid_data)
    
    def test_invalid_class_id_range_negative(self):
        """Test that negative class_id raises ValueError."""
        invalid_data = [
            (-1, np.full(297, 0.5, dtype=np.float32))
        ]
        
        with pytest.raises(ValueError, match="class_id must be in range"):
            aggregate_by_class(invalid_data)
    
    def test_invalid_class_id_range_too_large(self):
        """Test that class_id > 4 raises ValueError."""
        invalid_data = [
            (5, np.full(297, 0.5, dtype=np.float32))
        ]
        
        with pytest.raises(ValueError, match="class_id must be in range"):
            aggregate_by_class(invalid_data)
    
    def test_invalid_spectrum_type(self):
        """Test that non-array spectrum raises TypeError."""
        invalid_data = [
            (0, [0.5] * 297)
        ]
        
        with pytest.raises(TypeError, match="Spectrum must be a numpy array"):
            aggregate_by_class(invalid_data)
    
    def test_invalid_spectrum_shape(self):
        """Test that wrong shape spectrum raises ValueError."""
        invalid_data = [
            (0, np.full(100, 0.5, dtype=np.float32))
        ]
        
        with pytest.raises(ValueError, match="Spectrum must have shape"):
            aggregate_by_class(invalid_data)
    
    def test_missing_class_samples(self):
        """Test that missing samples for a class raises ValueError."""
        # Only provide samples for classes 0 and 1, missing 2, 3, 4
        incomplete_data = [
            (0, np.full(297, 0.3, dtype=np.float32)),
            (1, np.full(297, 0.5, dtype=np.float32))
        ]
        
        with pytest.raises(ValueError, match="No samples found for class"):
            aggregate_by_class(incomplete_data)


class TestAggregateByClassBasic:
    """Test basic aggregation functionality."""
    
    def test_single_sample_per_class(self):
        """Test aggregation with one sample per class."""
        # Create one sample for each class
        resampled = [
            (0, np.full(297, 0.1, dtype=np.float32)),
            (1, np.full(297, 0.2, dtype=np.float32)),
            (2, np.full(297, 0.3, dtype=np.float32)),
            (3, np.full(297, 0.4, dtype=np.float32)),
            (4, np.full(297, 0.5, dtype=np.float32))
        ]
        
        class_means = aggregate_by_class(resampled)
        
        # Verify output structure
        assert isinstance(class_means, dict)
        assert len(class_means) == 5
        assert set(class_means.keys()) == {0, 1, 2, 3, 4}
        
        # Verify each class mean
        for class_id in range(5):
            assert class_means[class_id].shape == (297,)
            assert class_means[class_id].dtype == np.float32
            expected_value = 0.1 * (class_id + 1)
            assert np.allclose(class_means[class_id], expected_value)
    
    def test_multiple_samples_per_class(self):
        """Test aggregation with multiple samples per class."""
        # Create multiple samples for each class
        resampled = [
            (0, np.full(297, 0.2, dtype=np.float32)),
            (0, np.full(297, 0.4, dtype=np.float32)),
            (1, np.full(297, 0.3, dtype=np.float32)),
            (1, np.full(297, 0.5, dtype=np.float32)),
            (1, np.full(297, 0.7, dtype=np.float32)),
            (2, np.full(297, 0.6, dtype=np.float32)),
            (3, np.full(297, 0.8, dtype=np.float32)),
            (4, np.full(297, 0.9, dtype=np.float32))
        ]
        
        class_means = aggregate_by_class(resampled)
        
        # Verify means
        assert np.allclose(class_means[0], 0.3)  # (0.2 + 0.4) / 2
        assert np.allclose(class_means[1], 0.5)  # (0.3 + 0.5 + 0.7) / 3
        assert np.allclose(class_means[2], 0.6)
        assert np.allclose(class_means[3], 0.8)
        assert np.allclose(class_means[4], 0.9)
    
    def test_varying_samples_per_class(self):
        """Test aggregation with different number of samples per class."""
        # Create varying number of samples
        resampled = [
            (0, np.full(297, 0.1, dtype=np.float32)),
            (0, np.full(297, 0.3, dtype=np.float32)),
            (0, np.full(297, 0.5, dtype=np.float32)),
            (1, np.full(297, 0.4, dtype=np.float32)),
            (1, np.full(297, 0.6, dtype=np.float32)),
            (2, np.full(297, 0.7, dtype=np.float32)),
            (3, np.full(297, 0.8, dtype=np.float32)),
            (3, np.full(297, 0.9, dtype=np.float32)),
            (3, np.full(297, 1.0, dtype=np.float32)),
            (3, np.full(297, 0.7, dtype=np.float32)),
            (4, np.full(297, 0.5, dtype=np.float32))
        ]
        
        class_means = aggregate_by_class(resampled)
        
        # Verify means
        assert np.allclose(class_means[0], 0.3)  # (0.1 + 0.3 + 0.5) / 3
        assert np.allclose(class_means[1], 0.5)  # (0.4 + 0.6) / 2
        assert np.allclose(class_means[2], 0.7)
        assert np.allclose(class_means[3], 0.85)  # (0.8 + 0.9 + 1.0 + 0.7) / 4
        assert np.allclose(class_means[4], 0.5)


class TestAggregateByClassNaNHandling:
    """Test NaN handling in aggregation."""
    
    def test_nan_values_ignored(self):
        """Test that NaN values are ignored in mean computation."""
        # Create samples with some NaN values
        spectrum1 = np.full(297, 0.3, dtype=np.float32)
        spectrum1[0] = np.nan  # First band is NaN
        
        spectrum2 = np.full(297, 0.5, dtype=np.float32)
        spectrum2[0] = np.nan  # First band is NaN
        
        spectrum3 = np.full(297, 0.7, dtype=np.float32)
        # Third spectrum has valid value for first band
        
        resampled = [
            (0, spectrum1),
            (0, spectrum2),
            (0, spectrum3),
            (1, np.full(297, 0.4, dtype=np.float32)),
            (2, np.full(297, 0.5, dtype=np.float32)),
            (3, np.full(297, 0.6, dtype=np.float32)),
            (4, np.full(297, 0.7, dtype=np.float32))
        ]
        
        class_means = aggregate_by_class(resampled)
        
        # For class 0, band 0 should be 0.7 (only valid value)
        assert np.isclose(class_means[0][0], 0.7)
        
        # Other bands should be mean of all three samples
        assert np.allclose(class_means[0][1:], 0.5)  # (0.3 + 0.5 + 0.7) / 3
    
    def test_all_nan_band(self):
        """Test that all-NaN band results in NaN mean."""
        # Create samples where band 0 is NaN for all samples in class 0
        spectrum1 = np.full(297, 0.3, dtype=np.float32)
        spectrum1[0] = np.nan
        
        spectrum2 = np.full(297, 0.5, dtype=np.float32)
        spectrum2[0] = np.nan
        
        resampled = [
            (0, spectrum1),
            (0, spectrum2),
            (1, np.full(297, 0.4, dtype=np.float32)),
            (2, np.full(297, 0.5, dtype=np.float32)),
            (3, np.full(297, 0.6, dtype=np.float32)),
            (4, np.full(297, 0.7, dtype=np.float32))
        ]
        
        class_means = aggregate_by_class(resampled)
        
        # Band 0 for class 0 should be NaN
        assert np.isnan(class_means[0][0])
        
        # Other bands should have valid means
        assert np.allclose(class_means[0][1:], 0.4)  # (0.3 + 0.5) / 2
    
    def test_partial_nan_bands(self):
        """Test handling of partial NaN values across bands."""
        # Create samples with NaN in different bands
        spectrum1 = np.full(297, 0.3, dtype=np.float32)
        spectrum1[0:10] = np.nan
        
        spectrum2 = np.full(297, 0.5, dtype=np.float32)
        spectrum2[5:15] = np.nan
        
        spectrum3 = np.full(297, 0.7, dtype=np.float32)
        spectrum3[10:20] = np.nan
        
        resampled = [
            (0, spectrum1),
            (0, spectrum2),
            (0, spectrum3),
            (1, np.full(297, 0.4, dtype=np.float32)),
            (2, np.full(297, 0.5, dtype=np.float32)),
            (3, np.full(297, 0.6, dtype=np.float32)),
            (4, np.full(297, 0.7, dtype=np.float32))
        ]
        
        class_means = aggregate_by_class(resampled)
        
        # Bands 0-4: spectrum1 is NaN, spectrum2 and spectrum3 valid -> mean = (0.5 + 0.7) / 2 = 0.6
        assert np.allclose(class_means[0][0:5], 0.6)
        
        # Bands 5-9: spectrum1 valid (0.3), spectrum2 is NaN, spectrum3 valid (0.7) -> mean = (0.3 + 0.7) / 2 = 0.5
        # Wait, let me recalculate: spectrum1[5:9] = 0.3, spectrum2[5:9] = NaN, spectrum3[5:9] = 0.7
        # Actually spectrum1[0:10] = NaN, so spectrum1[5:9] is also NaN!
        # So bands 5-9: spectrum1 is NaN, spectrum2 is NaN, spectrum3 valid (0.7) -> mean = 0.7
        assert np.allclose(class_means[0][5:10], 0.7)
        
        # Bands 10-14: spectrum1 valid (0.3), spectrum2 is NaN, spectrum3 is NaN -> mean = 0.3
        assert np.allclose(class_means[0][10:15], 0.3)
        
        # Bands 15-19: spectrum1 valid (0.3), spectrum2 valid (0.5), spectrum3 is NaN -> mean = (0.3 + 0.5) / 2 = 0.4
        assert np.allclose(class_means[0][15:20], 0.4)
        
        # Bands 20+: all three valid -> mean = (0.3 + 0.5 + 0.7) / 3 = 0.5
        assert np.allclose(class_means[0][20:], 0.5)
    
    def test_high_nan_percentage_warning(self, caplog):
        """Test that warning is logged for > 10% NaN bands."""
        # Create spectrum with > 10% NaN bands (> 30 bands)
        spectrum1 = np.full(297, 0.3, dtype=np.float32)
        spectrum1[0:50] = np.nan  # 50 NaN bands = 16.8%
        
        resampled = [
            (0, spectrum1),
            (1, np.full(297, 0.4, dtype=np.float32)),
            (2, np.full(297, 0.5, dtype=np.float32)),
            (3, np.full(297, 0.6, dtype=np.float32)),
            (4, np.full(297, 0.7, dtype=np.float32))
        ]
        
        with caplog.at_level("WARNING"):
            class_means = aggregate_by_class(resampled)
        
        # Check that warning was logged
        assert any("NaN bands" in record.message for record in caplog.records)
        assert any("16.8%" in record.message or "16.9%" in record.message 
                   for record in caplog.records)


class TestAggregateByClassIndependence:
    """Test that classes are processed independently."""
    
    def test_class_independence(self):
        """Test that aggregating one class doesn't affect others."""
        # Create initial data
        resampled1 = [
            (0, np.full(297, 0.2, dtype=np.float32)),
            (0, np.full(297, 0.4, dtype=np.float32)),
            (1, np.full(297, 0.5, dtype=np.float32)),
            (2, np.full(297, 0.6, dtype=np.float32)),
            (3, np.full(297, 0.7, dtype=np.float32)),
            (4, np.full(297, 0.8, dtype=np.float32))
        ]
        
        class_means1 = aggregate_by_class(resampled1)
        
        # Create data with additional samples for class 0
        resampled2 = [
            (0, np.full(297, 0.2, dtype=np.float32)),
            (0, np.full(297, 0.4, dtype=np.float32)),
            (0, np.full(297, 0.6, dtype=np.float32)),  # Additional sample
            (1, np.full(297, 0.5, dtype=np.float32)),
            (2, np.full(297, 0.6, dtype=np.float32)),
            (3, np.full(297, 0.7, dtype=np.float32)),
            (4, np.full(297, 0.8, dtype=np.float32))
        ]
        
        class_means2 = aggregate_by_class(resampled2)
        
        # Classes 1-4 should have same means in both cases
        for class_id in range(1, 5):
            assert np.allclose(class_means1[class_id], class_means2[class_id])
        
        # Class 0 should have different means
        assert not np.allclose(class_means1[0], class_means2[0])
        assert np.allclose(class_means1[0], 0.3)  # (0.2 + 0.4) / 2
        assert np.allclose(class_means2[0], 0.4)  # (0.2 + 0.4 + 0.6) / 3


class TestAggregateByClassRealistic:
    """Test with realistic spectral data."""
    
    def test_realistic_spectra(self):
        """Test aggregation with realistic varying spectra."""
        # Create realistic spectra with wavelength-dependent reflectance
        def create_spectrum(base_value, variation):
            """Create a spectrum with some variation."""
            spectrum = np.full(297, base_value, dtype=np.float32)
            # Add some wavelength-dependent variation
            for i in range(297):
                spectrum[i] += variation * np.sin(i / 30.0)
            return np.clip(spectrum, 0.0, 1.0)
        
        resampled = [
            (0, create_spectrum(0.3, 0.1)),
            (0, create_spectrum(0.35, 0.08)),
            (0, create_spectrum(0.32, 0.12)),
            (1, create_spectrum(0.5, 0.15)),
            (1, create_spectrum(0.48, 0.13)),
            (2, create_spectrum(0.6, 0.1)),
            (3, create_spectrum(0.4, 0.2)),
            (3, create_spectrum(0.45, 0.18)),
            (4, create_spectrum(0.7, 0.05))
        ]
        
        class_means = aggregate_by_class(resampled)
        
        # Verify output structure
        assert len(class_means) == 5
        for class_id in range(5):
            assert class_means[class_id].shape == (297,)
            assert class_means[class_id].dtype == np.float32
            # All values should be in valid reflectance range
            assert np.all(class_means[class_id] >= 0.0)
            assert np.all(class_means[class_id] <= 1.0)
    
    def test_mixed_nan_and_valid(self):
        """Test realistic scenario with mixed NaN and valid values."""
        # Simulate partial wavelength coverage for different samples
        spectrum1 = np.linspace(0.2, 0.8, 297, dtype=np.float32)
        spectrum1[0:20] = np.nan  # No coverage in short wavelengths
        
        spectrum2 = np.linspace(0.3, 0.7, 297, dtype=np.float32)
        spectrum2[250:297] = np.nan  # No coverage in long wavelengths
        
        spectrum3 = np.linspace(0.25, 0.75, 297, dtype=np.float32)
        # Full coverage
        
        resampled = [
            (0, spectrum1),
            (0, spectrum2),
            (0, spectrum3),
            (1, np.full(297, 0.4, dtype=np.float32)),
            (2, np.full(297, 0.5, dtype=np.float32)),
            (3, np.full(297, 0.6, dtype=np.float32)),
            (4, np.full(297, 0.7, dtype=np.float32))
        ]
        
        class_means = aggregate_by_class(resampled)
        
        # Verify class 0 has valid means
        assert class_means[0].shape == (297,)
        
        # Bands 0-19: only spectrum2 and spectrum3 valid
        assert not np.any(np.isnan(class_means[0][0:20]))
        
        # Bands 20-249: all three valid
        assert not np.any(np.isnan(class_means[0][20:250]))
        
        # Bands 250-296: only spectrum1 and spectrum3 valid
        assert not np.any(np.isnan(class_means[0][250:297]))
    
    def test_output_dtype_preserved(self):
        """Test that output dtype is float32."""
        resampled = [
            (0, np.full(297, 0.3, dtype=np.float32)),
            (1, np.full(297, 0.4, dtype=np.float32)),
            (2, np.full(297, 0.5, dtype=np.float32)),
            (3, np.full(297, 0.6, dtype=np.float32)),
            (4, np.full(297, 0.7, dtype=np.float32))
        ]
        
        class_means = aggregate_by_class(resampled)
        
        for class_id in range(5):
            assert class_means[class_id].dtype == np.float32
    
    def test_numpy_integer_class_id(self):
        """Test that numpy integer types work for class_id."""
        resampled = [
            (np.int32(0), np.full(297, 0.3, dtype=np.float32)),
            (np.int64(1), np.full(297, 0.4, dtype=np.float32)),
            (np.int16(2), np.full(297, 0.5, dtype=np.float32)),
            (np.int8(3), np.full(297, 0.6, dtype=np.float32)),
            (np.int_(4), np.full(297, 0.7, dtype=np.float32))
        ]
        
        class_means = aggregate_by_class(resampled)
        
        # Should work without errors
        assert len(class_means) == 5
        for class_id in range(5):
            expected_value = 0.3 + 0.1 * class_id
            assert np.allclose(class_means[class_id], expected_value)
