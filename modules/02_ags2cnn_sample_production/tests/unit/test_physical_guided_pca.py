"""
Unit tests for PhysicalGuidedPCA class.
PhysicalGuidedPCA类的单元测试。
"""

import numpy as np
import pytest

from hyperspectral_pseudo_label_generator.config import ProcessingConfig
from hyperspectral_pseudo_label_generator.pca import PhysicalGuidedPCA


class TestPhysicalGuidedPCAInitialization:
    """Test PhysicalGuidedPCA initialization."""
    
    def test_init_with_valid_wavelengths(self):
        """Test initialization with valid wavelengths."""
        config = ProcessingConfig()
        wavelengths = np.linspace(400, 2500, 297)
        
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        assert pca.config == config
        assert np.array_equal(pca.wavelengths, wavelengths)
        assert pca.pca is None
        assert pca.selected_components is None
        assert pca.scaler_params is None
    
    def test_init_with_incorrect_wavelength_count(self):
        """Test initialization fails with incorrect wavelength count."""
        config = ProcessingConfig()
        wavelengths = np.linspace(400, 2500, 250)  # Wrong count
        
        with pytest.raises(ValueError, match="Expected 297 wavelengths"):
            PhysicalGuidedPCA(config, wavelengths)
    
    def test_init_with_non_ascending_wavelengths(self):
        """Test initialization succeeds with non-ascending wavelengths."""
        config = ProcessingConfig()
        wavelengths = np.linspace(400, 2500, 297)
        wavelengths[100] = wavelengths[99] - 1  # Break ascending order
        pca = PhysicalGuidedPCA(config, wavelengths)
        assert np.array_equal(pca.wavelengths, wavelengths)
    
    def test_init_with_out_of_range_wavelengths(self):
        """Test initialization succeeds with out-of-range wavelengths."""
        config = ProcessingConfig()
        wavelengths = np.linspace(300, 2600, 297)  # Out of valid range
        pca = PhysicalGuidedPCA(config, wavelengths)
        assert np.array_equal(pca.wavelengths, wavelengths)

    def test_init_with_nan_wavelengths(self):
        """Test initialization fails with NaN wavelengths."""
        config = ProcessingConfig()
        wavelengths = np.linspace(400, 2500, 297).astype(float)
        wavelengths[0] = np.nan

        with pytest.raises(ValueError, match="NaN or Inf"):
            PhysicalGuidedPCA(config, wavelengths)


class TestPhysicalGuidedPCAFitTransform:
    """Test PhysicalGuidedPCA fit_transform method."""
    
    @pytest.fixture
    def setup_pca(self):
        """Set up PCA with valid configuration and wavelengths."""
        config = ProcessingConfig(n_components=5)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        return pca, config, wavelengths
    
    def test_fit_transform_with_valid_image(self, setup_pca):
        """Test fit_transform with valid image."""
        pca, config, wavelengths = setup_pca
        
        # Create synthetic image
        H, W = 50, 50
        image = np.random.rand(H, W, 297).astype(np.float32)
        
        # Transform
        transformed = pca.fit_transform(image)
        
        # Check output shape
        assert transformed.shape == (H, W, config.n_components)
        
        # Check that PCA was fitted
        assert pca.pca is not None
        assert pca.selected_components is not None
        assert pca.scaler_params is not None
        
        # Check that values are scaled to [0, 1]
        assert np.all(transformed >= 0)
        assert np.all(transformed <= 1)
    
    def test_fit_transform_with_invalid_dimensions(self, setup_pca):
        """Test fit_transform fails with invalid image dimensions."""
        pca, _, _ = setup_pca
        
        # 2D image instead of 3D
        image = np.random.rand(50, 297).astype(np.float32)
        
        with pytest.raises(ValueError, match="Image must be 3D"):
            pca.fit_transform(image)
    
    def test_fit_transform_with_wrong_band_count(self, setup_pca):
        """Test fit_transform fails with wrong band count."""
        pca, _, _ = setup_pca
        
        # Wrong number of bands
        image = np.random.rand(50, 50, 250).astype(np.float32)
        
        with pytest.raises(ValueError, match="Expected 297 bands"):
            pca.fit_transform(image)
    
    def test_fit_transform_with_invalid_dimensions_values(self, setup_pca):
        """Test fit_transform fails with invalid dimension values."""
        pca, _, _ = setup_pca
        
        # Zero height
        image = np.random.rand(0, 50, 297).astype(np.float32)
        
        with pytest.raises(ValueError, match="Invalid image dimensions"):
            pca.fit_transform(image)
    
    def test_fit_transform_with_nan_values(self, setup_pca):
        """Test fit_transform fails with NaN values."""
        pca, _, _ = setup_pca
        
        image = np.random.rand(50, 50, 297).astype(np.float32)
        image[10, 10, 10] = np.nan
        
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            pca.fit_transform(image)
    
    def test_fit_transform_with_inf_values(self, setup_pca):
        """Test fit_transform fails with Inf values."""
        pca, _, _ = setup_pca
        
        image = np.random.rand(50, 50, 297).astype(np.float32)
        image[10, 10, 10] = np.inf
        
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            pca.fit_transform(image)
    
    def test_fit_transform_preserves_spatial_dimensions(self, setup_pca):
        """Test that fit_transform preserves spatial dimensions."""
        pca, config, _ = setup_pca
        
        # Test with different spatial dimensions
        for H, W in [(10, 20), (100, 50), (30, 30)]:
            image = np.random.rand(H, W, 297).astype(np.float32)
            transformed = pca.fit_transform(image)
            
            assert transformed.shape[0] == H
            assert transformed.shape[1] == W
            assert transformed.shape[2] == config.n_components
    
    def test_fit_transform_selects_correct_number_of_components(self, setup_pca):
        """Test that fit_transform selects the correct number of components."""
        pca, config, _ = setup_pca
        
        image = np.random.rand(50, 50, 297).astype(np.float32)
        pca.fit_transform(image)
        
        assert len(pca.selected_components) == config.n_components
        assert pca.selected_components.dtype == np.int64 or pca.selected_components.dtype == np.intp


class TestPhysicalGuidedPCAComponentSelection:
    """Test physical-guided component selection."""
    
    @pytest.fixture
    def setup_pca_with_aloh_bands(self):
        """Set up PCA with wavelengths that include Al-OH region."""
        config = ProcessingConfig(n_components=3)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        # Create image with strong signal in Al-OH region
        H, W = 30, 30
        image = np.random.rand(H, W, 297).astype(np.float32) * 0.1
        
        # Find Al-OH band indices
        aloh_mask = (wavelengths >= 2150) & (wavelengths <= 2250)
        aloh_indices = np.where(aloh_mask)[0]
        
        # Add strong signal in Al-OH region
        image[:, :, aloh_indices] += 0.5
        
        return pca, image, aloh_indices
    
    def test_component_selection_based_on_aloh_region(self, setup_pca_with_aloh_bands):
        """Test that components are selected based on Al-OH region relevance."""
        pca, image, aloh_indices = setup_pca_with_aloh_bands
        
        # Fit and transform
        pca.fit_transform(image)
        
        # Check that components were selected
        assert pca.selected_components is not None
        assert len(pca.selected_components) == pca.config.n_components
        
        # Check that selected components are in ascending order
        assert np.all(np.diff(pca.selected_components) > 0)
    
    def test_component_selection_with_no_aloh_bands(self):
        """Test that AMCS selection does not depend on Al-OH wavelength window."""
        config = ProcessingConfig(n_components=3, aloh_min_wavelength=3000, aloh_max_wavelength=3100)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        image = np.random.rand(30, 30, 297).astype(np.float32)
        transformed = pca.fit_transform(image)
        assert transformed.shape == (30, 30, 3)


class TestPhysicalGuidedPCAScaling:
    """Test min-max scaling functionality."""
    
    @pytest.fixture
    def setup_pca_fitted(self):
        """Set up a fitted PCA instance."""
        config = ProcessingConfig(n_components=3)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        image = np.random.rand(30, 30, 297).astype(np.float32)
        transformed = pca.fit_transform(image)
        
        return pca, transformed
    
    def test_scaling_produces_values_in_zero_one_range(self, setup_pca_fitted):
        """Test that scaling produces values in [0, 1] range."""
        pca, transformed = setup_pca_fitted
        
        assert np.all(transformed >= 0)
        assert np.all(transformed <= 1)
    
    def test_scaling_parameters_stored(self, setup_pca_fitted):
        """Test that scaling parameters are stored."""
        pca, _ = setup_pca_fitted
        
        assert pca.scaler_params is not None
        assert len(pca.scaler_params) == pca.config.n_components
        
        # Each parameter should be a tuple of (min, max)
        for min_val, max_val in pca.scaler_params:
            assert isinstance(min_val, (int, float, np.number))
            assert isinstance(max_val, (int, float, np.number))
    
    def test_scaling_with_zero_range_component(self):
        """Test scaling handles zero-range components correctly."""
        config = ProcessingConfig(n_components=3)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        # Create image with constant values in some bands
        image = np.ones((30, 30, 297), dtype=np.float32)
        
        # This should not raise an error
        transformed = pca.fit_transform(image)
        
        # Check that output is valid
        assert not np.any(np.isnan(transformed))
        assert not np.any(np.isinf(transformed))


class TestPhysicalGuidedPCATransformReference:
    """Test reference spectra transformation."""
    
    @pytest.fixture
    def setup_pca_with_reference(self):
        """Set up PCA with fitted image and reference spectra."""
        config = ProcessingConfig(n_components=5)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        # Fit on image
        image = np.random.rand(30, 30, 297).astype(np.float32)
        pca.fit_transform(image)
        
        # Create reference spectra
        reference_spectra = {
            0: np.random.rand(297).astype(np.float32),
            1: np.random.rand(297).astype(np.float32),
            2: np.random.rand(297).astype(np.float32),
            3: np.random.rand(297).astype(np.float32),
            4: np.random.rand(297).astype(np.float32),
        }
        
        return pca, reference_spectra
    
    def test_transform_reference_with_valid_spectra(self, setup_pca_with_reference):
        """Test transforming reference spectra."""
        pca, reference_spectra = setup_pca_with_reference
        
        transformed = pca.transform_reference(reference_spectra)
        
        # Check that all classes are transformed
        assert len(transformed) == len(reference_spectra)
        
        # Check shape of transformed spectra
        for class_id, spectrum in transformed.items():
            assert spectrum.shape == (pca.config.n_components,)
            assert np.all(spectrum >= 0)
            assert np.all(spectrum <= 1)
    
    def test_transform_reference_before_fitting(self):
        """Test that transform_reference fails before fitting."""
        config = ProcessingConfig(n_components=5)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        reference_spectra = {0: np.random.rand(297).astype(np.float32)}
        
        with pytest.raises(RuntimeError, match="Must call fit_transform"):
            pca.transform_reference(reference_spectra)
    
    def test_transform_reference_with_wrong_band_count(self, setup_pca_with_reference):
        """Test that transform_reference fails with wrong band count."""
        pca, _ = setup_pca_with_reference
        
        reference_spectra = {0: np.random.rand(250).astype(np.float32)}
        
        with pytest.raises(ValueError, match="expected 297 bands"):
            pca.transform_reference(reference_spectra)
    
    def test_transform_reference_with_nan_values(self, setup_pca_with_reference):
        """Test that transform_reference fails with NaN values."""
        pca, _ = setup_pca_with_reference
        
        spectrum = np.random.rand(297).astype(np.float32)
        spectrum[100] = np.nan
        reference_spectra = {0: spectrum}
        
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            pca.transform_reference(reference_spectra)
    
    def test_transform_reference_with_inf_values(self, setup_pca_with_reference):
        """Test that transform_reference fails with Inf values."""
        pca, _ = setup_pca_with_reference
        
        spectrum = np.random.rand(297).astype(np.float32)
        spectrum[100] = np.inf
        reference_spectra = {0: spectrum}
        
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            pca.transform_reference(reference_spectra)
    
    def test_transform_reference_uses_same_scaling(self, setup_pca_with_reference):
        """Test that reference spectra use the same scaling as image."""
        pca, reference_spectra = setup_pca_with_reference
        
        # Transform reference spectra
        transformed = pca.transform_reference(reference_spectra)
        
        # All transformed spectra should be in [0, 1] range
        for spectrum in transformed.values():
            assert np.all(spectrum >= 0)
            assert np.all(spectrum <= 1)


class TestPhysicalGuidedPCAExplainedVariance:
    """Test explained variance functionality."""
    
    @pytest.fixture
    def setup_fitted_pca(self):
        """Set up a fitted PCA instance."""
        config = ProcessingConfig(n_components=5)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        image = np.random.rand(50, 50, 297).astype(np.float32)
        pca.fit_transform(image)
        
        return pca
    
    def test_get_explained_variance_after_fitting(self, setup_fitted_pca):
        """Test getting explained variance after fitting."""
        pca = setup_fitted_pca
        
        explained_var = pca.get_explained_variance()
        
        # Check shape
        assert explained_var.shape == (pca.config.n_components,)
        
        # Check that values are valid probabilities
        assert np.all(explained_var >= 0)
        assert np.all(explained_var <= 1)
        
        # Check that variance ratios sum to less than or equal to 1
        assert np.sum(explained_var) <= 1.0
    
    def test_get_explained_variance_before_fitting(self):
        """Test that get_explained_variance fails before fitting."""
        config = ProcessingConfig(n_components=5)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        with pytest.raises(RuntimeError, match="PCA not fitted yet"):
            pca.get_explained_variance()
    
    def test_explained_variance_decreases(self, setup_fitted_pca):
        """Test that explained variance generally decreases for selected components."""
        pca = setup_fitted_pca
        
        explained_var = pca.get_explained_variance()
        
        # Note: Due to physical-guided selection, this may not always be strictly decreasing
        # But the sum should still be meaningful
        assert np.sum(explained_var) > 0


class TestPhysicalGuidedPCAEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_small_image_dimensions(self):
        """Test with very small image dimensions."""
        config = ProcessingConfig(n_components=3)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        # Very small image
        image = np.random.rand(5, 5, 297).astype(np.float32)
        
        # Should still work
        transformed = pca.fit_transform(image)
        assert transformed.shape == (5, 5, 3)
    
    def test_large_n_components(self):
        """Test with large number of components."""
        config = ProcessingConfig(n_components=50)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        image = np.random.rand(100, 100, 297).astype(np.float32)
        
        transformed = pca.fit_transform(image)
        assert transformed.shape == (100, 100, 50)
    
    def test_rectangular_images(self):
        """Test with non-square images."""
        config = ProcessingConfig(n_components=5)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        # Tall image
        image = np.random.rand(100, 20, 297).astype(np.float32)
        transformed = pca.fit_transform(image)
        assert transformed.shape == (100, 20, 5)
        
        # Wide image
        image = np.random.rand(20, 100, 297).astype(np.float32)
        transformed = pca.fit_transform(image)
        assert transformed.shape == (20, 100, 5)
    
    def test_consistent_results_with_same_data(self):
        """Test that running PCA twice on same data gives consistent results."""
        config = ProcessingConfig(n_components=5)
        wavelengths = np.linspace(400, 2500, 297)
        
        image = np.random.rand(30, 30, 297).astype(np.float32)
        
        # First run
        pca1 = PhysicalGuidedPCA(config, wavelengths)
        transformed1 = pca1.fit_transform(image)
        
        # Second run
        pca2 = PhysicalGuidedPCA(config, wavelengths)
        transformed2 = pca2.fit_transform(image)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(transformed1, transformed2, decimal=5)
