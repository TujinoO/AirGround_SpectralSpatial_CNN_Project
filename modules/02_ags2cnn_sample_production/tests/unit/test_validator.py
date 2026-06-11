"""
Unit tests for InputValidator class.
InputValidator类的单元测试。
"""

import pytest
import numpy as np
from hyperspectral_pseudo_label_generator.input.validator import InputValidator
from hyperspectral_pseudo_label_generator.config import ProcessingConfig


class TestValidateImage:
    """Tests for validate_image method."""
    
    def test_valid_image(self):
        """Test that a valid image passes validation."""
        image = np.random.rand(100, 100, 297)
        # Should not raise any exception
        InputValidator.validate_image(image)
    
    def test_invalid_dimensions_2d(self):
        """Test that 2D image raises ValueError."""
        image = np.random.rand(100, 297)
        with pytest.raises(ValueError, match="Image must be 3D, got 2D"):
            InputValidator.validate_image(image)
    
    def test_invalid_dimensions_4d(self):
        """Test that 4D image raises ValueError."""
        image = np.random.rand(10, 100, 100, 297)
        with pytest.raises(ValueError, match="Image must be 3D, got 4D"):
            InputValidator.validate_image(image)
    
    def test_invalid_band_count_too_few(self):
        """Test that image with too few bands raises ValueError."""
        image = np.random.rand(100, 100, 200)
        with pytest.raises(ValueError, match="Expected 297 bands, got 200"):
            InputValidator.validate_image(image)
    
    def test_invalid_band_count_too_many(self):
        """Test that image with too many bands raises ValueError."""
        image = np.random.rand(100, 100, 300)
        with pytest.raises(ValueError, match="Expected 297 bands, got 300"):
            InputValidator.validate_image(image)
    
    def test_invalid_height_zero(self):
        """Test that image with zero height raises ValueError."""
        image = np.random.rand(0, 100, 297)
        with pytest.raises(ValueError, match="Invalid image dimensions"):
            InputValidator.validate_image(image)
    
    def test_invalid_width_zero(self):
        """Test that image with zero width raises ValueError."""
        image = np.random.rand(100, 0, 297)
        with pytest.raises(ValueError, match="Invalid image dimensions"):
            InputValidator.validate_image(image)
    
    def test_image_contains_nan(self):
        """Test that image with NaN values raises ValueError."""
        image = np.random.rand(100, 100, 297)
        image[50, 50, 100] = np.nan
        with pytest.raises(ValueError, match="Image contains NaN or Inf values"):
            InputValidator.validate_image(image)
    
    def test_image_contains_inf(self):
        """Test that image with Inf values raises ValueError."""
        image = np.random.rand(100, 100, 297)
        image[50, 50, 100] = np.inf
        with pytest.raises(ValueError, match="Image contains NaN or Inf values"):
            InputValidator.validate_image(image)
    
    def test_image_contains_negative_inf(self):
        """Test that image with -Inf values raises ValueError."""
        image = np.random.rand(100, 100, 297)
        image[50, 50, 100] = -np.inf
        with pytest.raises(ValueError, match="Image contains NaN or Inf values"):
            InputValidator.validate_image(image)
    
    def test_small_image(self):
        """Test that small valid image passes validation."""
        image = np.random.rand(1, 1, 297)
        # Should not raise any exception
        InputValidator.validate_image(image)
    
    def test_large_image(self):
        """Test that large valid image passes validation."""
        image = np.random.rand(1000, 1000, 297)
        # Should not raise any exception
        InputValidator.validate_image(image)


class TestValidateReferenceSpectra:
    """Tests for validate_reference_spectra method."""
    
    def test_valid_spectra(self):
        """Test that valid reference spectra pass validation."""
        spectra = {i: np.random.rand(297) for i in range(5)}
        # Should not raise any exception
        InputValidator.validate_reference_spectra(spectra)
    
    def test_invalid_class_count_too_few(self):
        """Test that too few classes raises ValueError."""
        spectra = {i: np.random.rand(297) for i in range(3)}
        with pytest.raises(ValueError, match="Expected 5 classes, got 3"):
            InputValidator.validate_reference_spectra(spectra)
    
    def test_invalid_class_count_too_many(self):
        """Test that too many classes raises ValueError."""
        spectra = {i: np.random.rand(297) for i in range(7)}
        with pytest.raises(ValueError, match="Expected 5 classes, got 7"):
            InputValidator.validate_reference_spectra(spectra)
    
    def test_invalid_band_count_in_class(self):
        """Test that spectrum with wrong band count raises ValueError."""
        spectra = {i: np.random.rand(297) for i in range(5)}
        spectra[2] = np.random.rand(200)  # Wrong band count
        with pytest.raises(ValueError, match="Class 2: expected 297 bands, got 200"):
            InputValidator.validate_reference_spectra(spectra)
    
    def test_spectrum_contains_nan(self):
        """Test that spectrum with NaN values raises ValueError."""
        spectra = {i: np.random.rand(297) for i in range(5)}
        spectra[3][100] = np.nan
        with pytest.raises(ValueError, match="Class 3: contains NaN or Inf values"):
            InputValidator.validate_reference_spectra(spectra)
    
    def test_spectrum_contains_inf(self):
        """Test that spectrum with Inf values raises ValueError."""
        spectra = {i: np.random.rand(297) for i in range(5)}
        spectra[1][50] = np.inf
        with pytest.raises(ValueError, match="Class 1: contains NaN or Inf values"):
            InputValidator.validate_reference_spectra(spectra)
    
    def test_spectrum_contains_negative_inf(self):
        """Test that spectrum with -Inf values raises ValueError."""
        spectra = {i: np.random.rand(297) for i in range(5)}
        spectra[4][200] = -np.inf
        with pytest.raises(ValueError, match="Class 4: contains NaN or Inf values"):
            InputValidator.validate_reference_spectra(spectra)
    
    def test_multiple_invalid_spectra(self):
        """Test that first invalid spectrum is caught."""
        spectra = {i: np.random.rand(297) for i in range(5)}
        spectra[0][10] = np.nan
        spectra[1][20] = np.inf
        # Should catch the first invalid one (class 0)
        with pytest.raises(ValueError, match="Class 0: contains NaN or Inf values"):
            InputValidator.validate_reference_spectra(spectra)
    
    def test_empty_spectra_dict(self):
        """Test that empty spectra dictionary raises ValueError."""
        spectra = {}
        with pytest.raises(ValueError, match="Expected 5 classes, got 0"):
            InputValidator.validate_reference_spectra(spectra)


class TestValidateWavelengths:
    """Tests for validate_wavelengths method."""
    
    def test_valid_wavelengths(self):
        """Test that valid wavelengths pass validation."""
        wavelengths = np.linspace(400, 2500, 297)
        # Should not raise any exception
        InputValidator.validate_wavelengths(wavelengths)
    
    def test_invalid_count_too_few(self):
        """Test that too few wavelengths raises ValueError."""
        wavelengths = np.linspace(400, 2500, 200)
        with pytest.raises(ValueError, match="Expected 297 wavelengths, got 200"):
            InputValidator.validate_wavelengths(wavelengths)
    
    def test_invalid_count_too_many(self):
        """Test that too many wavelengths raises ValueError."""
        wavelengths = np.linspace(400, 2500, 300)
        with pytest.raises(ValueError, match="Expected 297 wavelengths, got 300"):
            InputValidator.validate_wavelengths(wavelengths)
    
    def test_non_ascending_wavelengths_pass(self):
        """Test that non-ascending wavelengths pass validation."""
        wavelengths = np.linspace(400, 2500, 297)
        wavelengths[100], wavelengths[101] = wavelengths[101], wavelengths[100]
        InputValidator.validate_wavelengths(wavelengths)

    def test_out_of_range_wavelengths_pass(self):
        """Test that out-of-range wavelengths pass validation."""
        wavelengths = np.linspace(300, 2600, 297)
        InputValidator.validate_wavelengths(wavelengths)

    def test_wavelengths_contains_nan(self):
        """Test that wavelengths containing NaN raises ValueError."""
        wavelengths = np.linspace(400, 2500, 297).astype(float)
        wavelengths[0] = np.nan
        with pytest.raises(ValueError, match="Wavelengths contains NaN or Inf values"):
            InputValidator.validate_wavelengths(wavelengths)

    def test_wavelengths_contains_inf(self):
        """Test that wavelengths containing Inf raises ValueError."""
        wavelengths = np.linspace(400, 2500, 297).astype(float)
        wavelengths[0] = np.inf
        with pytest.raises(ValueError, match="Wavelengths contains NaN or Inf values"):
            InputValidator.validate_wavelengths(wavelengths)


class TestValidateConfig:
    """Tests for validate_config method."""
    
    def test_valid_config_default(self):
        """Test that default configuration passes validation."""
        config = ProcessingConfig()
        # Should not raise any exception
        InputValidator.validate_config(config)
    
    def test_valid_config_custom(self):
        """Test that custom valid configuration passes validation."""
        config = ProcessingConfig(
            n_components=10,
            ore_percentile=5.0,
            non_ore_percentile=10.0,
            ambiguity_threshold=0.2,
            epsilon=1e-8,
            chunk_size=500
        )
        # Should not raise any exception
        InputValidator.validate_config(config)
    
    def test_invalid_n_components_zero(self):
        """Test that n_components=0 raises ValueError."""
        config = ProcessingConfig(n_components=0)
        with pytest.raises(ValueError, match="n_components must be positive"):
            InputValidator.validate_config(config)
    
    def test_invalid_n_components_negative(self):
        """Test that negative n_components raises ValueError."""
        config = ProcessingConfig(n_components=-5)
        with pytest.raises(ValueError, match="n_components must be positive"):
            InputValidator.validate_config(config)
    
    def test_invalid_ore_percentile_zero(self):
        """Test that ore_percentile=0 raises ValueError."""
        config = ProcessingConfig(ore_percentile=0)
        with pytest.raises(ValueError, match="ore_percentile must be between 0 and 100"):
            InputValidator.validate_config(config)
    
    def test_invalid_ore_percentile_100(self):
        """Test that ore_percentile=100 raises ValueError."""
        config = ProcessingConfig(ore_percentile=100)
        with pytest.raises(ValueError, match="ore_percentile must be between 0 and 100"):
            InputValidator.validate_config(config)
    
    def test_invalid_ore_percentile_negative(self):
        """Test that negative ore_percentile raises ValueError."""
        config = ProcessingConfig(ore_percentile=-5)
        with pytest.raises(ValueError, match="ore_percentile must be between 0 and 100"):
            InputValidator.validate_config(config)
    
    def test_invalid_non_ore_percentile_zero(self):
        """Test that non_ore_percentile=0 raises ValueError."""
        config = ProcessingConfig(non_ore_percentile=0)
        with pytest.raises(ValueError, match="non_ore_percentile must be between 0 and 100"):
            InputValidator.validate_config(config)
    
    def test_invalid_non_ore_percentile_100(self):
        """Test that non_ore_percentile=100 raises ValueError."""
        config = ProcessingConfig(non_ore_percentile=100)
        with pytest.raises(ValueError, match="non_ore_percentile must be between 0 and 100"):
            InputValidator.validate_config(config)
    
    def test_invalid_ambiguity_threshold_negative(self):
        """Test that negative ambiguity_threshold raises ValueError."""
        config = ProcessingConfig(ambiguity_threshold=-0.1)
        with pytest.raises(ValueError, match="ambiguity_threshold must be non-negative"):
            InputValidator.validate_config(config)
    
    def test_valid_ambiguity_threshold_zero(self):
        """Test that ambiguity_threshold=0 passes validation."""
        config = ProcessingConfig(ambiguity_threshold=0)
        # Should not raise any exception
        InputValidator.validate_config(config)
    
    def test_invalid_epsilon_zero(self):
        """Test that epsilon=0 raises ValueError."""
        config = ProcessingConfig(epsilon=0)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            InputValidator.validate_config(config)
    
    def test_invalid_epsilon_negative(self):
        """Test that negative epsilon raises ValueError."""
        config = ProcessingConfig(epsilon=-1e-10)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            InputValidator.validate_config(config)
    
    def test_invalid_chunk_size_zero(self):
        """Test that chunk_size=0 raises ValueError."""
        config = ProcessingConfig(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            InputValidator.validate_config(config)
    
    def test_invalid_chunk_size_negative(self):
        """Test that negative chunk_size raises ValueError."""
        config = ProcessingConfig(chunk_size=-100)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            InputValidator.validate_config(config)
    
    def test_invalid_aloh_wavelength_range(self):
        """Test that invalid Al-OH wavelength range raises ValueError."""
        config = ProcessingConfig(aloh_min_wavelength=2250, aloh_max_wavelength=2150)
        with pytest.raises(ValueError, match="aloh_min_wavelength must be less than aloh_max_wavelength"):
            InputValidator.validate_config(config)
    
    def test_invalid_aloh_wavelength_out_of_range(self):
        """Test that Al-OH wavelengths outside valid range raise ValueError."""
        config = ProcessingConfig(aloh_min_wavelength=300, aloh_max_wavelength=2200)
        with pytest.raises(ValueError, match="Al-OH wavelength range must be within \\[400, 2500\\] nm"):
            InputValidator.validate_config(config)
    
    def test_invalid_config_type_none(self):
        """Test that None config raises TypeError."""
        with pytest.raises(TypeError, match="Expected ProcessingConfig instance, got NoneType"):
            InputValidator.validate_config(None)
    
    def test_invalid_config_type_dict(self):
        """Test that dict config raises TypeError."""
        config_dict = {"n_components": 5}
        with pytest.raises(TypeError, match="Expected ProcessingConfig instance, got dict"):
            InputValidator.validate_config(config_dict)
    
    def test_invalid_config_type_string(self):
        """Test that string config raises TypeError."""
        with pytest.raises(TypeError, match="Expected ProcessingConfig instance, got str"):
            InputValidator.validate_config("config")
    
    def test_boundary_values(self):
        """Test configuration with boundary values."""
        config = ProcessingConfig(
            n_components=1,
            ore_percentile=0.1,
            non_ore_percentile=99.9,
            ambiguity_threshold=0.0,
            epsilon=1e-15,
            chunk_size=1,
            aloh_min_wavelength=400,
            aloh_max_wavelength=2500
        )
        # Should not raise any exception
        InputValidator.validate_config(config)
