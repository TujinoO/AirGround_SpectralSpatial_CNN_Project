"""
Unit tests for the filters module.
滤波器模块的单元测试
"""

import numpy as np
import pytest
from gsrsl_pipeline.data_models import GroundSpectrum
from gsrsl_pipeline.filters import apply_savgol_filter


class TestApplySavgolFilter:
    """Test suite for apply_savgol_filter function."""
    
    def test_basic_filtering(self):
        """
        Test that basic filtering works and produces smoothed output.
        测试基本滤波功能并产生平滑输出
        """
        # Create a noisy spectrum
        wavelengths = np.linspace(350, 2500, 500, dtype=np.float64)
        # Create a smooth signal with added noise
        clean_signal = np.sin(wavelengths / 200) * 0.2 + 0.5
        noise = np.random.RandomState(42).normal(0, 0.02, len(wavelengths))
        noisy_reflectances = clean_signal + noise
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=noisy_reflectances,
            class_id=0,
            filename="test_noisy.csv",
            is_anomalous=False
        )
        
        # Apply filter
        filtered = apply_savgol_filter(spectrum)
        
        # Verify wavelengths are unchanged (Requirement 4.2)
        assert np.array_equal(filtered.wavelengths, spectrum.wavelengths)
        
        # Verify reflectances are smoothed (lower standard deviation)
        assert np.std(filtered.reflectances) < np.std(spectrum.reflectances)
        
        # Verify reflectances are in valid range (Requirement 4.3)
        assert np.all(filtered.reflectances >= 0.0)
        assert np.all(filtered.reflectances <= 1.0)
        
        # Verify other attributes are preserved
        assert filtered.class_id == spectrum.class_id
        assert filtered.filename == spectrum.filename
        assert filtered.is_anomalous == spectrum.is_anomalous
    
    def test_wavelengths_unchanged(self):
        """
        Test that wavelength array is preserved unchanged (Requirement 4.2).
        测试波长数组保持不变（需求4.2）
        """
        wavelengths = np.linspace(400, 2400, 300, dtype=np.float64)
        reflectances = np.random.RandomState(42).uniform(0.1, 0.9, 300)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=1,
            filename="test_wavelengths.csv",
            is_anomalous=False
        )
        
        filtered = apply_savgol_filter(spectrum)
        
        # Wavelengths should be exactly the same
        assert np.array_equal(filtered.wavelengths, spectrum.wavelengths)
        # Should be a copy, not the same object
        assert filtered.wavelengths is not spectrum.wavelengths
    
    def test_reflectance_range_preserved(self):
        """
        Test that filtered reflectances remain in [0.0, 1.0] range (Requirement 4.3).
        测试滤波后的反射率保持在[0.0, 1.0]范围内（需求4.3）
        """
        wavelengths = np.linspace(350, 2500, 400, dtype=np.float64)
        # Create reflectances near boundaries
        reflectances = np.concatenate([
            np.full(100, 0.05),  # Near lower bound
            np.full(100, 0.5),   # Middle
            np.full(100, 0.95),  # Near upper bound
            np.random.RandomState(42).uniform(0.0, 1.0, 100)  # Random
        ])
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=2,
            filename="test_range.csv",
            is_anomalous=False
        )
        
        filtered = apply_savgol_filter(spectrum)
        
        # All values must be in valid range
        assert np.all(filtered.reflectances >= 0.0), \
            f"Found reflectances < 0.0: {filtered.reflectances[filtered.reflectances < 0.0]}"
        assert np.all(filtered.reflectances <= 1.0), \
            f"Found reflectances > 1.0: {filtered.reflectances[filtered.reflectances > 1.0]}"
    
    def test_exactly_15_points(self):
        """
        Test edge case: spectrum with exactly 15 points (minimum for window_length=15).
        测试边界情况：恰好15个数据点的光谱（window_length=15的最小值）
        """
        wavelengths = np.linspace(350, 364, 15, dtype=np.float64)
        reflectances = np.random.RandomState(42).uniform(0.2, 0.8, 15)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=3,
            filename="test_15points.csv",
            is_anomalous=False
        )
        
        # Should work without error
        filtered = apply_savgol_filter(spectrum)
        
        assert len(filtered.wavelengths) == 15
        assert len(filtered.reflectances) == 15
        assert np.all(filtered.reflectances >= 0.0)
        assert np.all(filtered.reflectances <= 1.0)
    
    def test_fewer_than_15_points_raises_error(self):
        """
        Test error case: spectrum with fewer than 15 points (Requirement 4.4).
        测试错误情况：少于15个数据点的光谱（需求4.4）
        """
        wavelengths = np.linspace(350, 363, 14, dtype=np.float64)
        reflectances = np.random.RandomState(42).uniform(0.2, 0.8, 14)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=4,
            filename="test_14points.csv",
            is_anomalous=False
        )
        
        # Should raise ValueError with descriptive message
        with pytest.raises(ValueError) as exc_info:
            apply_savgol_filter(spectrum)
        
        error_msg = str(exc_info.value)
        assert "14 data points" in error_msg
        assert "15 points" in error_msg
        assert "window_length" in error_msg
    
    def test_custom_window_length(self):
        """
        Test that custom window_length parameter works.
        测试自定义window_length参数有效
        """
        wavelengths = np.linspace(350, 2500, 300, dtype=np.float64)
        reflectances = np.random.RandomState(42).uniform(0.2, 0.8, 300)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=0,
            filename="test_custom_window.csv",
            is_anomalous=False
        )
        
        # Test with different window lengths
        filtered_11 = apply_savgol_filter(spectrum, window_length=11, polyorder=3)
        filtered_21 = apply_savgol_filter(spectrum, window_length=21, polyorder=3)
        
        # Both should work
        assert len(filtered_11.reflectances) == 300
        assert len(filtered_21.reflectances) == 300
        
        # Larger window should produce smoother result
        # (though this is not always guaranteed for all signals)
        assert np.all(filtered_11.reflectances >= 0.0)
        assert np.all(filtered_21.reflectances >= 0.0)
    
    def test_custom_polyorder(self):
        """
        Test that custom polyorder parameter works.
        测试自定义polyorder参数有效
        """
        wavelengths = np.linspace(350, 2500, 300, dtype=np.float64)
        reflectances = np.random.RandomState(42).uniform(0.2, 0.8, 300)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=1,
            filename="test_custom_poly.csv",
            is_anomalous=False
        )
        
        # Test with different polynomial orders
        filtered_2 = apply_savgol_filter(spectrum, window_length=15, polyorder=2)
        filtered_4 = apply_savgol_filter(spectrum, window_length=15, polyorder=4)
        
        # Both should work
        assert len(filtered_2.reflectances) == 300
        assert len(filtered_4.reflectances) == 300
        assert np.all(filtered_2.reflectances >= 0.0)
        assert np.all(filtered_4.reflectances >= 0.0)
    
    def test_even_window_length_raises_error(self):
        """
        Test that even window_length raises ValueError.
        测试偶数window_length引发ValueError
        """
        wavelengths = np.linspace(350, 2500, 300, dtype=np.float64)
        reflectances = np.random.RandomState(42).uniform(0.2, 0.8, 300)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=2,
            filename="test_even_window.csv",
            is_anomalous=False
        )
        
        # Even window_length should raise error
        with pytest.raises(ValueError) as exc_info:
            apply_savgol_filter(spectrum, window_length=14, polyorder=3)
        
        error_msg = str(exc_info.value)
        assert "must be odd" in error_msg
        assert "14" in error_msg
    
    def test_polyorder_too_large_raises_error(self):
        """
        Test that polyorder >= window_length raises ValueError.
        测试polyorder >= window_length引发ValueError
        """
        wavelengths = np.linspace(350, 2500, 300, dtype=np.float64)
        reflectances = np.random.RandomState(42).uniform(0.2, 0.8, 300)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=3,
            filename="test_large_poly.csv",
            is_anomalous=False
        )
        
        # polyorder too large for window_length
        with pytest.raises(ValueError) as exc_info:
            apply_savgol_filter(spectrum, window_length=15, polyorder=14)
        
        error_msg = str(exc_info.value)
        assert "window_length" in error_msg
        assert "polyorder" in error_msg
    
    def test_preserves_anomalous_flag(self):
        """
        Test that is_anomalous flag is preserved.
        测试is_anomalous标志被保留
        """
        wavelengths = np.linspace(350, 2500, 300, dtype=np.float64)
        reflectances = np.random.RandomState(42).uniform(0.2, 0.8, 300)
        
        # Create anomalous spectrum
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=4,
            filename="test_anomalous.csv",
            is_anomalous=True  # Marked as anomalous
        )
        
        filtered = apply_savgol_filter(spectrum)
        
        # Anomalous flag should be preserved
        assert filtered.is_anomalous == True
    
    def test_preserves_class_id(self):
        """
        Test that class_id is preserved for all classes.
        测试所有类别的class_id被保留
        """
        wavelengths = np.linspace(350, 2500, 300, dtype=np.float64)
        reflectances = np.random.RandomState(42).uniform(0.2, 0.8, 300)
        
        for class_id in range(5):
            spectrum = GroundSpectrum(
                wavelengths=wavelengths,
                reflectances=reflectances,
                class_id=class_id,
                filename=f"test_class_{class_id}.csv",
                is_anomalous=False
            )
            
            filtered = apply_savgol_filter(spectrum)
            assert filtered.class_id == class_id
    
    def test_realistic_spectrum(self):
        """
        Test with a realistic spectrum that mimics actual spectral data.
        测试模拟实际光谱数据的真实光谱
        """
        # Create realistic wavelength range (1nm resolution)
        wavelengths = np.arange(350, 2501, 1, dtype=np.float64)
        
        # Create realistic reflectance with absorption features
        # Simulate mineral absorption features around 1400nm and 2200nm
        reflectances = 0.5 + 0.2 * np.sin(wavelengths / 300)
        reflectances -= 0.15 * np.exp(-((wavelengths - 1400) ** 2) / (2 * 50 ** 2))
        reflectances -= 0.2 * np.exp(-((wavelengths - 2200) ** 2) / (2 * 80 ** 2))
        
        # Add realistic noise
        noise = np.random.RandomState(42).normal(0, 0.01, len(wavelengths))
        reflectances += noise
        
        # Clip to valid range
        reflectances = np.clip(reflectances, 0.0, 1.0)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=0,
            filename="realistic_spectrum.csv",
            is_anomalous=False
        )
        
        # Apply filter with default parameters
        filtered = apply_savgol_filter(spectrum)
        
        # Verify basic properties
        assert len(filtered.reflectances) == len(wavelengths)
        assert np.all(filtered.reflectances >= 0.0)
        assert np.all(filtered.reflectances <= 1.0)
        
        # Verify smoothing occurred
        assert np.std(filtered.reflectances) < np.std(spectrum.reflectances)
        
        # Verify absorption features are still present (not over-smoothed)
        # Find minimum around 2200nm
        idx_2200 = np.argmin(np.abs(wavelengths - 2200))
        window = slice(idx_2200 - 50, idx_2200 + 50)
        
        # There should still be a local minimum near 2200nm
        local_min_idx = np.argmin(filtered.reflectances[window])
        assert abs(local_min_idx - 50) < 20, "Absorption feature was over-smoothed"
    
    def test_dtype_preservation(self):
        """
        Test that output reflectances have correct dtype (float64).
        测试输出反射率具有正确的dtype（float64）
        """
        wavelengths = np.linspace(350, 2500, 300, dtype=np.float64)
        reflectances = np.random.RandomState(42).uniform(0.2, 0.8, 300)
        
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=reflectances,
            class_id=0,
            filename="test_dtype.csv",
            is_anomalous=False
        )
        
        filtered = apply_savgol_filter(spectrum)
        
        # Check dtypes
        assert filtered.wavelengths.dtype == np.float64
        assert filtered.reflectances.dtype == np.float64
