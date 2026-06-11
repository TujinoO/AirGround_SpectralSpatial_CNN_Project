"""
Integration tests for MetadataParser with InputValidator.
MetadataParser与InputValidator的集成测试。
"""

import pytest
import numpy as np
import json
import tempfile
import os
from hyperspectral_pseudo_label_generator.input import MetadataParser, InputValidator


class TestMetadataParserValidatorIntegration:
    """Integration tests for MetadataParser with InputValidator."""
    
    def test_parse_and_validate_valid_wavelengths(self):
        """Test parsing and validating valid wavelengths."""
        # Create valid wavelengths (297 bands, ascending, in valid range)
        wavelengths_data = list(np.linspace(400, 2500, 297))
        metadata = {'wavelengths': wavelengths_data}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            temp_path = f.name
        
        try:
            # Parse wavelengths
            wavelengths = MetadataParser.parse_wavelengths(temp_path)
            
            # Validate - should not raise
            InputValidator.validate_wavelengths(wavelengths)
            
            # Verify we can identify Al-OH band
            mask, indices = MetadataParser.identify_aloh_band(wavelengths)
            assert len(indices) > 0
        finally:
            os.unlink(temp_path)
    
    def test_parse_and_validate_wrong_number_of_bands(self):
        """Test that validator catches wrong number of bands."""
        # Create wavelengths with wrong number of bands
        wavelengths_data = list(np.linspace(400, 2500, 200))
        metadata = {'wavelengths': wavelengths_data}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            temp_path = f.name
        
        try:
            # Parse wavelengths
            wavelengths = MetadataParser.parse_wavelengths(temp_path)
            
            # Validate - should raise
            with pytest.raises(ValueError, match="Expected 297 wavelengths"):
                InputValidator.validate_wavelengths(wavelengths)
        finally:
            os.unlink(temp_path)
    
    def test_parse_and_validate_non_ascending_wavelengths(self):
        """Test that non-ascending wavelengths pass validation."""
        # Create non-ascending wavelengths
        wavelengths_data = list(np.linspace(400, 2500, 297))
        wavelengths_data[100], wavelengths_data[101] = wavelengths_data[101], wavelengths_data[100]
        metadata = {'wavelengths': wavelengths_data}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            temp_path = f.name
        
        try:
            # Parse wavelengths
            wavelengths = MetadataParser.parse_wavelengths(temp_path)
            
            # Validate - should not raise
            InputValidator.validate_wavelengths(wavelengths)
        finally:
            os.unlink(temp_path)
    
    def test_parse_and_validate_out_of_range_wavelengths(self):
        """Test that out-of-range wavelengths pass validation."""
        # Create wavelengths outside valid range
        wavelengths_data = list(np.linspace(300, 2600, 297))
        metadata = {'wavelengths': wavelengths_data}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            temp_path = f.name
        
        try:
            # Parse wavelengths
            wavelengths = MetadataParser.parse_wavelengths(temp_path)
            
            # Validate - should not raise
            InputValidator.validate_wavelengths(wavelengths)
        finally:
            os.unlink(temp_path)

    def test_txt_format_integration(self):
        """Test integration with GF5A-style .txt metadata format."""
        lines = [
            "Wavelengths 1 = 387.21",
            "FWHM 1 = 4.38",
            "Wavelengths 2 = 391.49",
            "FWHM 2 = 4.38",
        ]
        for i in range(3, 298):
            lines.append(f"Wavelengths {i} = {400.0 + i:.2f}")
            lines.append(f"FWHM {i} = 8.25")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("\n".join(lines))
            temp_path = f.name

        try:
            wavelengths = MetadataParser.parse_wavelengths(temp_path)
            InputValidator.validate_wavelengths(wavelengths)
            assert wavelengths.shape == (297,)
        finally:
            os.unlink(temp_path)
    
    def test_aloh_identification_with_valid_gf5_wavelengths(self):
        """Test Al-OH band identification with realistic GF-5 wavelengths."""
        # Create realistic GF-5 wavelengths
        wavelengths_data = list(np.linspace(450, 2450, 297))
        metadata = {'wavelengths': wavelengths_data}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            temp_path = f.name
        
        try:
            # Parse wavelengths
            wavelengths = MetadataParser.parse_wavelengths(temp_path)
            
            # Validate
            InputValidator.validate_wavelengths(wavelengths)
            
            # Identify Al-OH band
            mask, indices = MetadataParser.identify_aloh_band(wavelengths)
            
            # Verify Al-OH band is identified
            assert len(indices) > 0
            
            # Verify identified wavelengths are in correct range
            aloh_wavelengths = wavelengths[indices]
            assert np.all(aloh_wavelengths >= 2150.0)
            assert np.all(aloh_wavelengths <= 2250.0)
            
            # Verify reasonable number of bands in Al-OH region
            # For 297 bands covering 450-2450nm, expect ~15 bands in 100nm range
            assert 5 <= len(indices) <= 30
        finally:
            os.unlink(temp_path)
    
    def test_csv_format_integration(self):
        """Test integration with CSV format."""
        # Create valid wavelengths in CSV format
        wavelengths_data = np.linspace(400, 2500, 297)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            np.savetxt(f, wavelengths_data, delimiter=',')
            temp_path = f.name
        
        try:
            # Parse wavelengths
            wavelengths = MetadataParser.parse_wavelengths(temp_path)
            
            # Validate
            InputValidator.validate_wavelengths(wavelengths)
            
            # Identify Al-OH band
            mask, indices = MetadataParser.identify_aloh_band(wavelengths)
            assert len(indices) > 0
        finally:
            os.unlink(temp_path)
