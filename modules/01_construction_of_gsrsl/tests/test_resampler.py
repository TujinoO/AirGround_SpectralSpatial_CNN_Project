"""
Unit tests for spectral resampler module.
光谱重采样器模块的单元测试
"""

import pytest
import numpy as np
from gsrsl_pipeline.resampler import SpectralResampler
from gsrsl_pipeline.data_models import GroundSpectrum


class TestSpectralResamplerInit:
    """Test SpectralResampler initialization."""
    
    def test_valid_initialization(self):
        """Test initialization with valid inputs."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        assert resampler.sat_wavelengths.shape == (297,)
        assert resampler.sat_fwhms.shape == (297,)
        assert resampler.sat_sigmas.shape == (297,)
        assert np.allclose(resampler.sat_sigmas, sat_fwhms / 2.355, rtol=1e-3)
    
    def test_invalid_wavelengths_type(self):
        """Test that non-array wavelengths raise TypeError."""
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        with pytest.raises(TypeError, match="sat_wavelengths must be a numpy array"):
            SpectralResampler([400, 500, 600], sat_fwhms)
    
    def test_invalid_wavelengths_shape(self):
        """Test that wrong shape wavelengths raise ValueError."""
        sat_wavelengths = np.linspace(400, 2400, 100, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        with pytest.raises(ValueError, match="sat_wavelengths must have shape"):
            SpectralResampler(sat_wavelengths, sat_fwhms)
    
    def test_invalid_fwhms_type(self):
        """Test that non-array FWHMs raise TypeError."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        
        with pytest.raises(TypeError, match="sat_fwhms must be a numpy array"):
            SpectralResampler(sat_wavelengths, [6.0] * 297)
    
    def test_invalid_fwhms_shape(self):
        """Test that wrong shape FWHMs raise ValueError."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(100, 6.0, dtype=np.float64)
        
        with pytest.raises(ValueError, match="sat_fwhms must have shape"):
            SpectralResampler(sat_wavelengths, sat_fwhms)
    
    def test_negative_wavelengths(self):
        """Test that negative wavelengths raise ValueError."""
        sat_wavelengths = np.linspace(-100, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        with pytest.raises(ValueError, match="sat_wavelengths must contain only positive values"):
            SpectralResampler(sat_wavelengths, sat_fwhms)
    
    def test_negative_fwhms(self):
        """Test that negative FWHMs raise ValueError."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, -6.0, dtype=np.float64)
        
        with pytest.raises(ValueError, match="sat_fwhms must contain only positive values"):
            SpectralResampler(sat_wavelengths, sat_fwhms)


class TestComputeSigmas:
    """Test sigma computation from FWHM."""
    
    def test_sigma_computation_formula(self):
        """Test that sigma = FWHM / 2.355 approximately."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.array([4.0, 5.0, 6.0, 7.0, 8.0] * 59 + [4.0, 5.0], dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        # Verify sigma ≈ FWHM / 2.355
        expected_sigmas = sat_fwhms / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        assert np.allclose(resampler.sat_sigmas, expected_sigmas, rtol=1e-10)
    
    def test_sigma_values_positive(self):
        """Test that all sigma values are positive."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.linspace(4.0, 8.0, 297, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        assert np.all(resampler.sat_sigmas > 0)
    
    def test_sigma_proportional_to_fwhm(self):
        """Test that sigma is proportional to FWHM."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.linspace(4.0, 8.0, 297, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        # Ratio should be constant
        ratios = sat_fwhms / resampler.sat_sigmas
        assert np.allclose(ratios, ratios[0], rtol=1e-10)


class TestGaussianWeight:
    """Test Gaussian weight function."""
    
    def test_weight_at_center(self):
        """Test that weight is 1.0 at band center."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        center = 1000.0
        sigma = 2.5
        weight = resampler._gaussian_weight(center, center, sigma)
        
        assert np.isclose(weight, 1.0, rtol=1e-10)
    
    def test_weight_decreases_with_distance(self):
        """Test that weight decreases as distance from center increases."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        center = 1000.0
        sigma = 2.5
        
        weight_0 = resampler._gaussian_weight(center, center, sigma)
        weight_1 = resampler._gaussian_weight(center + sigma, center, sigma)
        weight_2 = resampler._gaussian_weight(center + 2*sigma, center, sigma)
        weight_3 = resampler._gaussian_weight(center + 3*sigma, center, sigma)
        
        assert weight_0 > weight_1 > weight_2 > weight_3
    
    def test_weight_symmetric(self):
        """Test that weight is symmetric around center."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        center = 1000.0
        sigma = 2.5
        offset = 5.0
        
        weight_plus = resampler._gaussian_weight(center + offset, center, sigma)
        weight_minus = resampler._gaussian_weight(center - offset, center, sigma)
        
        assert np.isclose(weight_plus, weight_minus, rtol=1e-10)
    
    def test_weight_range(self):
        """Test that weight is always in range [0, 1]."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        center = 1000.0
        sigma = 2.5
        
        # Test various wavelengths
        for wavelength in np.linspace(900, 1100, 50):
            weight = resampler._gaussian_weight(wavelength, center, sigma)
            assert 0.0 <= weight <= 1.0


class TestResample:
    """Test spectral resampling."""
    
    def test_resample_output_shape(self):
        """Test that resampled output has shape (297,)."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        # Create ground spectrum with full coverage
        ground_wavelengths = np.linspace(350, 2500, 2151, dtype=np.float64)
        ground_reflectances = np.full(2151, 0.5, dtype=np.float64)
        spectrum = GroundSpectrum(
            wavelengths=ground_wavelengths,
            reflectances=ground_reflectances,
            class_id=0,
            filename="test.csv",
            is_anomalous=False
        )
        
        resampled = resampler.resample(spectrum)
        
        assert resampled.shape == (297,)
        assert resampled.dtype == np.float32
    
    def test_resample_constant_spectrum(self):
        """Test resampling a constant spectrum."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        # Create constant ground spectrum
        ground_wavelengths = np.linspace(350, 2500, 2151, dtype=np.float64)
        ground_reflectances = np.full(2151, 0.5, dtype=np.float64)
        spectrum = GroundSpectrum(
            wavelengths=ground_wavelengths,
            reflectances=ground_reflectances,
            class_id=0,
            filename="test.csv",
            is_anomalous=False
        )
        
        resampled = resampler.resample(spectrum)
        
        # For constant spectrum, all bands should have same value
        valid_bands = ~np.isnan(resampled)
        assert np.allclose(resampled[valid_bands], 0.5, rtol=1e-5)
    
    def test_resample_no_overlap(self):
        """Test resampling with no overlap produces NaN."""
        # Satellite bands in visible range
        sat_wavelengths = np.linspace(400, 700, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        # Ground spectrum in NIR range (no overlap)
        ground_wavelengths = np.linspace(1000, 2500, 1501, dtype=np.float64)
        ground_reflectances = np.full(1501, 0.5, dtype=np.float64)
        spectrum = GroundSpectrum(
            wavelengths=ground_wavelengths,
            reflectances=ground_reflectances,
            class_id=0,
            filename="test.csv",
            is_anomalous=False
        )
        
        resampled = resampler.resample(spectrum)
        
        # All bands should be NaN due to no overlap
        assert np.all(np.isnan(resampled))
    
    def test_resample_partial_overlap(self):
        """Test resampling with partial overlap."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        # Ground spectrum with limited range
        ground_wavelengths = np.linspace(500, 1500, 1001, dtype=np.float64)
        ground_reflectances = np.full(1001, 0.5, dtype=np.float64)
        spectrum = GroundSpectrum(
            wavelengths=ground_wavelengths,
            reflectances=ground_reflectances,
            class_id=0,
            filename="test.csv",
            is_anomalous=False
        )
        
        resampled = resampler.resample(spectrum)
        
        # Some bands should have valid values, some should be NaN
        assert np.any(~np.isnan(resampled))
        assert np.any(np.isnan(resampled))
    
    def test_resample_invalid_input_type(self):
        """Test that invalid input type raises TypeError."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        with pytest.raises(TypeError, match="spectrum must be a GroundSpectrum object"):
            resampler.resample("not a spectrum")
    
    def test_resample_preserves_reflectance_range(self):
        """Test that resampled values stay in valid reflectance range."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        # Create ground spectrum with values in [0, 1]
        ground_wavelengths = np.linspace(350, 2500, 2151, dtype=np.float64)
        ground_reflectances = np.random.uniform(0.0, 1.0, 2151).astype(np.float64)
        spectrum = GroundSpectrum(
            wavelengths=ground_wavelengths,
            reflectances=ground_reflectances,
            class_id=0,
            filename="test.csv",
            is_anomalous=False
        )
        
        resampled = resampler.resample(spectrum)
        
        # All valid values should be in [0, 1]
        valid_values = resampled[~np.isnan(resampled)]
        assert np.all(valid_values >= 0.0)
        assert np.all(valid_values <= 1.0)
    
    def test_resample_linear_spectrum(self):
        """Test resampling a linear spectrum."""
        sat_wavelengths = np.linspace(400, 2400, 297, dtype=np.float64)
        sat_fwhms = np.full(297, 6.0, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        # Create linear ground spectrum: reflectance increases with wavelength
        ground_wavelengths = np.linspace(350, 2500, 2151, dtype=np.float64)
        ground_reflectances = np.linspace(0.0, 1.0, 2151, dtype=np.float64)
        spectrum = GroundSpectrum(
            wavelengths=ground_wavelengths,
            reflectances=ground_reflectances,
            class_id=0,
            filename="test.csv",
            is_anomalous=False
        )
        
        resampled = resampler.resample(spectrum)
        
        # Resampled values should also increase (monotonic)
        valid_bands = ~np.isnan(resampled)
        valid_resampled = resampled[valid_bands]
        
        # Check that values generally increase
        # (allowing for small numerical variations)
        diffs = np.diff(valid_resampled)
        assert np.sum(diffs > 0) > 0.9 * len(diffs)


class TestResampleIntegration:
    """Integration tests for resampling with realistic data."""
    
    def test_resample_realistic_spectrum(self):
        """Test resampling with realistic GF-5 specifications."""
        # Realistic GF-5 band specifications (simplified)
        sat_wavelengths = np.linspace(450, 2450, 297, dtype=np.float64)
        sat_fwhms = np.linspace(4.0, 8.0, 297, dtype=np.float64)
        
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        # Realistic ground spectrum from TSG8
        ground_wavelengths = np.linspace(350, 2500, 2151, dtype=np.float64)
        # Simulate a typical mineral spectrum with absorption features
        ground_reflectances = 0.3 + 0.2 * np.sin(ground_wavelengths / 200.0)
        ground_reflectances = np.clip(ground_reflectances, 0.0, 1.0).astype(np.float64)
        
        spectrum = GroundSpectrum(
            wavelengths=ground_wavelengths,
            reflectances=ground_reflectances,
            class_id=0,
            filename="realistic_sample.csv",
            is_anomalous=False
        )
        
        resampled = resampler.resample(spectrum)
        
        # Verify output properties
        assert resampled.shape == (297,)
        assert resampled.dtype == np.float32
        
        # Most bands should have valid values
        valid_bands = ~np.isnan(resampled)
        assert np.sum(valid_bands) > 250  # At least 250 out of 297 bands
        
        # Valid values should be in reflectance range
        valid_values = resampled[valid_bands]
        assert np.all(valid_values >= 0.0)
        assert np.all(valid_values <= 1.0)
