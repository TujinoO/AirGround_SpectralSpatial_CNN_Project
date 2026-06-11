"""
Unit tests for MetadataParser class.
MetadataParser类的单元测试。
"""

import pytest
import numpy as np
import json
import tempfile
import os
from hyperspectral_pseudo_label_generator.input.metadata_parser import MetadataParser


class TestParseWavelengths:
    """Tests for parse_wavelengths method."""
    
    def test_parse_json_format(self):
        """Test parsing wavelengths from JSON format."""
        # Create temporary JSON file
        wavelengths_data = list(np.linspace(400, 2500, 297))
        metadata = {'wavelengths': wavelengths_data}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            temp_path = f.name
        
        try:
            # Parse wavelengths
            result = MetadataParser.parse_wavelengths(temp_path)
            
            # Verify shape
            assert result.shape == (297,)
            
            # Verify values
            expected = np.array(wavelengths_data)
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            os.unlink(temp_path)
    
    def test_parse_csv_format(self):
        """Test parsing wavelengths from CSV format."""
        # Create temporary CSV file
        wavelengths_data = np.linspace(400, 2500, 297)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            np.savetxt(f, wavelengths_data, delimiter=',')
            temp_path = f.name
        
        try:
            # Parse wavelengths
            result = MetadataParser.parse_wavelengths(temp_path)
            
            # Verify shape
            assert result.shape == (297,)
            
            # Verify values
            np.testing.assert_array_almost_equal(result, wavelengths_data)
        finally:
            os.unlink(temp_path)
    
    def test_unsupported_format_raises_error(self):
        """Test that unsupported file format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported metadata format"):
            MetadataParser.parse_wavelengths("metadata.dat")
    
    def test_parse_json_with_nested_structure(self):
        """Test parsing wavelengths from JSON with nested structure."""
        # Create temporary JSON file with nested structure
        wavelengths_data = list(np.linspace(400, 2500, 297))
        metadata = {
            'sensor': 'GF-5',
            'wavelengths': wavelengths_data,
            'other_data': {'key': 'value'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            temp_path = f.name
        
        try:
            # Parse wavelengths
            result = MetadataParser.parse_wavelengths(temp_path)
            
            # Verify shape
            assert result.shape == (297,)
            
            # Verify values
            expected = np.array(wavelengths_data)
            np.testing.assert_array_almost_equal(result, expected)
        finally:
            os.unlink(temp_path)


class TestIdentifyAlohBand:
    """Tests for identify_aloh_band method."""
    
    def test_identify_aloh_band_default_range(self):
        """Test Al-OH band identification with default range (2150-2250nm)."""
        # Create wavelengths covering the full GF-5 range
        wavelengths = np.linspace(400, 2500, 297)
        
        # Identify Al-OH band
        mask, indices = MetadataParser.identify_aloh_band(wavelengths)
        
        # Verify mask is boolean array with correct shape
        assert mask.dtype == bool
        assert mask.shape == (297,)
        
        # Verify indices are within valid range
        assert len(indices) > 0
        assert np.all(indices >= 0)
        assert np.all(indices < 297)
        
        # Verify wavelengths in identified region are within Al-OH range
        aloh_wavelengths = wavelengths[indices]
        assert np.all(aloh_wavelengths >= 2150.0)
        assert np.all(aloh_wavelengths <= 2250.0)
        
        # Verify mask and indices are consistent
        np.testing.assert_array_equal(mask, np.isin(np.arange(297), indices))
    
    def test_identify_aloh_band_custom_range(self):
        """Test Al-OH band identification with custom range."""
        wavelengths = np.linspace(400, 2500, 297)
        
        # Use custom range
        min_wl = 2100.0
        max_wl = 2300.0
        
        mask, indices = MetadataParser.identify_aloh_band(
            wavelengths, 
            min_wavelength=min_wl, 
            max_wavelength=max_wl
        )
        
        # Verify wavelengths in identified region are within custom range
        aloh_wavelengths = wavelengths[indices]
        assert np.all(aloh_wavelengths >= min_wl)
        assert np.all(aloh_wavelengths <= max_wl)
    
    def test_identify_aloh_band_no_wavelengths_in_range(self):
        """Test that error is raised when no wavelengths in Al-OH range."""
        # Create wavelengths that don't cover Al-OH range
        wavelengths = np.linspace(400, 1000, 297)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="No wavelengths found in Al-OH absorption region"):
            MetadataParser.identify_aloh_band(wavelengths)
    
    def test_identify_aloh_band_edge_cases(self):
        """Test Al-OH band identification with edge case wavelengths."""
        # Create wavelengths with exact boundaries
        wavelengths = np.array([2149.9, 2150.0, 2200.0, 2250.0, 2250.1])
        
        mask, indices = MetadataParser.identify_aloh_band(wavelengths)
        
        # Should include 2150.0, 2200.0, 2250.0 but not 2149.9 or 2250.1
        expected_indices = np.array([1, 2, 3])
        np.testing.assert_array_equal(indices, expected_indices)
        
        # Verify mask
        expected_mask = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(mask, expected_mask)
    
    def test_identify_aloh_band_single_wavelength(self):
        """Test Al-OH band identification with single wavelength in range."""
        # Create wavelengths with only one in Al-OH range
        wavelengths = np.array([400, 1000, 2200, 2300, 2500])
        
        mask, indices = MetadataParser.identify_aloh_band(wavelengths)
        
        # Should identify only the one wavelength
        assert len(indices) == 1
        assert indices[0] == 2
        assert wavelengths[indices[0]] == 2200
    
    def test_identify_aloh_band_returns_sorted_indices(self):
        """Test that returned indices are sorted."""
        wavelengths = np.linspace(400, 2500, 297)
        
        mask, indices = MetadataParser.identify_aloh_band(wavelengths)
        
        # Verify indices are sorted
        assert np.all(np.diff(indices) > 0)
    
    def test_identify_aloh_band_mask_consistency(self):
        """Test that mask and indices are consistent."""
        wavelengths = np.linspace(400, 2500, 297)
        
        mask, indices = MetadataParser.identify_aloh_band(wavelengths)
        
        # Verify that mask[i] is True if and only if i is in indices
        for i in range(len(wavelengths)):
            if mask[i]:
                assert i in indices
            else:
                assert i not in indices
    
    def test_identify_aloh_band_error_message_includes_range(self):
        """Test that error message includes wavelength range information."""
        wavelengths = np.linspace(400, 1000, 297)
        
        with pytest.raises(ValueError) as exc_info:
            MetadataParser.identify_aloh_band(wavelengths)
        
        error_msg = str(exc_info.value)
        # Should include the Al-OH range
        assert "2150" in error_msg or "2250" in error_msg
        # Should include the actual wavelength range
        assert "400" in error_msg or "1000" in error_msg
