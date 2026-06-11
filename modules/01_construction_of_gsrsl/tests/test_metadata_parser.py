"""
Unit tests for metadata parser module
元数据解析器模块的单元测试
"""

import pytest
import numpy as np
import tempfile
import os
from gsrsl_pipeline.metadata_parser import parse_metadata


class TestMetadataParser:
    """Test suite for parse_metadata function"""
    
    def test_parse_valid_metadata_file(self):
        """Test parsing a valid GF-5 metadata file with 297 bands"""
        # Create a temporary metadata file with 297 bands
        content = []
        for i in range(1, 298):
            wavelength = 387.21 + (i - 1) * 5.0  # Mock wavelengths
            fwhm = 4.38 + (i - 1) * 0.01  # Mock FWHM values
            content.append(f"Wavelengths {i} = {wavelength:.2f}")
            content.append(f"FWHM {i} = {fwhm:.2f}")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('\n'.join(content))
            temp_path = f.name
        
        try:
            wavelengths, fwhms = parse_metadata(temp_path)
            
            # Verify output types and shapes
            assert isinstance(wavelengths, np.ndarray)
            assert isinstance(fwhms, np.ndarray)
            assert wavelengths.shape == (297,)
            assert fwhms.shape == (297,)
            assert wavelengths.dtype == np.float64
            assert fwhms.dtype == np.float64
            
            # Verify first and last values
            assert wavelengths[0] == pytest.approx(387.21, rel=1e-5)
            assert wavelengths[296] == pytest.approx(387.21 + 296 * 5.0, rel=1e-5)
            assert fwhms[0] == pytest.approx(4.38, rel=1e-5)
            assert fwhms[296] == pytest.approx(4.38 + 296 * 0.01, rel=1e-5)
        finally:
            os.unlink(temp_path)
    
    def test_parse_metadata_with_exactly_297_bands(self):
        """Test edge case: exactly 297 bands"""
        content = []
        for i in range(1, 298):
            content.append(f"Wavelengths {i} = {400.0 + i}")
            content.append(f"FWHM {i} = {5.0}")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('\n'.join(content))
            temp_path = f.name
        
        try:
            wavelengths, fwhms = parse_metadata(temp_path)
            assert wavelengths.shape == (297,)
            assert fwhms.shape == (297,)
        finally:
            os.unlink(temp_path)
    
    def test_parse_metadata_insufficient_wavelength_bands(self):
        """Test error case: fewer than 297 wavelength bands"""
        content = []
        for i in range(1, 297):  # Only 296 bands
            content.append(f"Wavelengths {i} = {400.0 + i}")
            content.append(f"FWHM {i} = {5.0}")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('\n'.join(content))
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="insufficient wavelength bands"):
                parse_metadata(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_parse_metadata_insufficient_fwhm_bands(self):
        """Test error case: fewer than 297 FWHM bands"""
        content = []
        for i in range(1, 298):
            content.append(f"Wavelengths {i} = {400.0 + i}")
        for i in range(1, 297):  # Only 296 FWHM bands
            content.append(f"FWHM {i} = {5.0}")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('\n'.join(content))
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="insufficient FWHM bands"):
                parse_metadata(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_parse_metadata_non_numeric_wavelength(self):
        """Test error case: non-numeric wavelength values"""
        content = []
        for i in range(1, 298):
            if i == 100:
                content.append(f"Wavelengths {i} = invalid")
            else:
                content.append(f"Wavelengths {i} = {400.0 + i}")
            content.append(f"FWHM {i} = {5.0}")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('\n'.join(content))
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="insufficient wavelength bands"):
                parse_metadata(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_parse_metadata_non_numeric_fwhm(self):
        """Test error case: non-numeric FWHM values"""
        content = []
        for i in range(1, 298):
            content.append(f"Wavelengths {i} = {400.0 + i}")
            if i == 100:
                content.append(f"FWHM {i} = invalid")
            else:
                content.append(f"FWHM {i} = {5.0}")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('\n'.join(content))
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="insufficient FWHM bands"):
                parse_metadata(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_parse_metadata_file_not_found(self):
        """Test error case: metadata file doesn't exist"""
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            parse_metadata("/nonexistent/path/to/metadata.txt")
    
    def test_parse_metadata_with_extra_whitespace(self):
        """Test parsing with various whitespace patterns"""
        content = []
        for i in range(1, 298):
            # Add varying whitespace
            content.append(f"Wavelengths   {i}   =   {400.0 + i}")
            content.append(f"FWHM\t{i}\t=\t{5.0}")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('\n'.join(content))
            temp_path = f.name
        
        try:
            wavelengths, fwhms = parse_metadata(temp_path)
            assert wavelengths.shape == (297,)
            assert fwhms.shape == (297,)
        finally:
            os.unlink(temp_path)
