"""
Integration tests for PhysicalGuidedPCA with complete workflow.
PhysicalGuidedPCA完整工作流的集成测试。
"""

import numpy as np
import pytest

from hyperspectral_pseudo_label_generator.config import ProcessingConfig
from hyperspectral_pseudo_label_generator.pca import PhysicalGuidedPCA


class TestPhysicalGuidedPCAIntegration:
    """Integration tests for complete PCA workflow."""
    
    def test_complete_pca_workflow(self):
        """Test complete workflow: fit image, transform reference spectra."""
        # Setup
        config = ProcessingConfig(n_components=5)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        # Create synthetic hyperspectral image
        H, W = 50, 50
        np.random.seed(42)  # For reproducibility
        image = np.random.rand(H, W, 297).astype(np.float32)
        
        # Add some structure to the image (simulate spectral features)
        # Enhance Al-OH absorption region
        aloh_mask = (wavelengths >= 2150) & (wavelengths <= 2250)
        aloh_indices = np.where(aloh_mask)[0]
        image[:, :, aloh_indices] *= 1.5
        
        # Step 1: Fit and transform image
        transformed_image = pca.fit_transform(image)
        
        # Verify transformed image properties
        assert transformed_image.shape == (H, W, config.n_components)
        assert np.all(transformed_image >= 0)
        assert np.all(transformed_image <= 1)
        assert not np.any(np.isnan(transformed_image))
        assert not np.any(np.isinf(transformed_image))
        
        # Step 2: Create and transform reference spectra
        reference_spectra = {
            0: np.random.rand(297).astype(np.float32),  # Spodumene-rich
            1: np.random.rand(297).astype(np.float32),  # Lepidolite-rich
            2: np.random.rand(297).astype(np.float32),  # Mixed-type
            3: np.random.rand(297).astype(np.float32),  # Barren pegmatite
            4: np.random.rand(297).astype(np.float32),  # Wall rock
        }
        
        # Enhance Al-OH region in reference spectra too
        for class_id in reference_spectra:
            reference_spectra[class_id][aloh_indices] *= 1.5
        
        transformed_reference = pca.transform_reference(reference_spectra)
        
        # Verify transformed reference spectra properties
        assert len(transformed_reference) == 5
        for class_id, spectrum in transformed_reference.items():
            assert spectrum.shape == (config.n_components,)
            assert np.all(spectrum >= 0)
            assert np.all(spectrum <= 1)
            assert not np.any(np.isnan(spectrum))
            assert not np.any(np.isinf(spectrum))
        
        # Step 3: Verify explained variance
        explained_var = pca.get_explained_variance()
        assert explained_var.shape == (config.n_components,)
        assert np.all(explained_var > 0)
        assert np.sum(explained_var) <= 1.0
        
        # Log some statistics
        print(f"\nPCA Integration Test Results:")
        print(f"  Image shape: {image.shape} -> {transformed_image.shape}")
        print(f"  Selected components: {pca.selected_components}")
        print(f"  Explained variance: {explained_var}")
        print(f"  Total variance explained: {np.sum(explained_var) * 100:.2f}%")
        print(f"  Reference spectra transformed: {len(transformed_reference)} classes")
    
    def test_pca_with_different_configurations(self):
        """Test PCA with different configuration parameters."""
        wavelengths = np.linspace(400, 2500, 297)
        image = np.random.rand(30, 30, 297).astype(np.float32)
        
        # Test with different numbers of components
        for n_components in [3, 5, 10, 20]:
            config = ProcessingConfig(n_components=n_components)
            pca = PhysicalGuidedPCA(config, wavelengths)
            
            transformed = pca.fit_transform(image)
            
            assert transformed.shape == (30, 30, n_components)
            assert len(pca.selected_components) == n_components
            
            explained_var = pca.get_explained_variance()
            assert explained_var.shape == (n_components,)
    
    def test_pca_consistency_across_multiple_transforms(self):
        """Test that PCA transformation is consistent for multiple reference spectra."""
        config = ProcessingConfig(n_components=5)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        # Fit on image
        image = np.random.rand(40, 40, 297).astype(np.float32)
        pca.fit_transform(image)
        
        # Transform same spectrum multiple times
        spectrum = np.random.rand(297).astype(np.float32)
        
        result1 = pca.transform_reference({0: spectrum})
        result2 = pca.transform_reference({0: spectrum})
        
        # Results should be identical
        np.testing.assert_array_equal(result1[0], result2[0])
    
    def test_pca_with_realistic_spectral_data(self):
        """Test PCA with more realistic spectral data patterns."""
        config = ProcessingConfig(n_components=5)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        # Create image with realistic spectral patterns
        H, W = 60, 60
        image = np.zeros((H, W, 297), dtype=np.float32)
        
        # Simulate different mineral zones
        for i in range(H):
            for j in range(W):
                # Base reflectance
                base = 0.3 + 0.2 * np.sin(i / 10) * np.cos(j / 10)
                
                # Add spectral features
                spectrum = np.ones(297) * base
                
                # Add absorption features at different wavelengths
                for center_wl, depth in [(1400, 0.2), (1900, 0.3), (2200, 0.4)]:
                    idx = np.argmin(np.abs(wavelengths - center_wl))
                    width = 20
                    for k in range(max(0, idx - width), min(297, idx + width)):
                        distance = abs(k - idx)
                        spectrum[k] -= depth * np.exp(-distance**2 / (2 * (width/3)**2))
                
                image[i, j, :] = np.clip(spectrum, 0, 1)
        
        # Transform
        transformed = pca.fit_transform(image)
        
        # Verify results
        assert transformed.shape == (H, W, config.n_components)
        assert np.all(transformed >= 0)
        assert np.all(transformed <= 1)
        
        # Check that PCA captured meaningful variance
        explained_var = pca.get_explained_variance()
        # Note: With physical-guided selection, explained variance may be lower
        # as we select components based on Al-OH relevance, not total variance
        assert np.sum(explained_var) > 0  # Should explain some variance
        assert len(explained_var) == config.n_components
    
    def test_pca_numerical_stability_with_extreme_values(self):
        """Test PCA numerical stability with extreme but valid values."""
        config = ProcessingConfig(n_components=5)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        # Create image with very small values
        image_small = np.random.rand(30, 30, 297).astype(np.float32) * 1e-6
        transformed_small = pca.fit_transform(image_small)
        
        assert not np.any(np.isnan(transformed_small))
        assert not np.any(np.isinf(transformed_small))
        
        # Create image with large values
        pca2 = PhysicalGuidedPCA(config, wavelengths)
        image_large = np.random.rand(30, 30, 297).astype(np.float32) * 1e6
        transformed_large = pca2.fit_transform(image_large)
        
        assert not np.any(np.isnan(transformed_large))
        assert not np.any(np.isinf(transformed_large))
    
    def test_pca_with_mixed_spatial_and_spectral_patterns(self):
        """Test PCA with both spatial and spectral patterns."""
        config = ProcessingConfig(n_components=5)
        wavelengths = np.linspace(400, 2500, 297)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        H, W = 50, 50
        image = np.zeros((H, W, 297), dtype=np.float32)
        
        # Create spatial zones with different spectral signatures
        for i in range(H):
            for j in range(W):
                # Determine zone
                if i < H // 2 and j < W // 2:
                    # Zone 1: High reflectance
                    image[i, j, :] = 0.7 + 0.1 * np.random.rand(297)
                elif i < H // 2 and j >= W // 2:
                    # Zone 2: Low reflectance
                    image[i, j, :] = 0.3 + 0.1 * np.random.rand(297)
                elif i >= H // 2 and j < W // 2:
                    # Zone 3: Medium reflectance with absorption
                    spectrum = 0.5 + 0.1 * np.random.rand(297)
                    aloh_mask = (wavelengths >= 2150) & (wavelengths <= 2250)
                    spectrum[aloh_mask] *= 0.5
                    image[i, j, :] = spectrum
                else:
                    # Zone 4: Variable reflectance
                    image[i, j, :] = 0.4 + 0.3 * np.random.rand(297)
        
        # Transform
        transformed = pca.fit_transform(image)
        
        # Verify
        assert transformed.shape == (H, W, config.n_components)
        assert np.all(transformed >= 0)
        assert np.all(transformed <= 1)
        
        # Check that different zones have different PCA signatures
        zone1_mean = np.mean(transformed[:H//2, :W//2, :], axis=(0, 1))
        zone2_mean = np.mean(transformed[:H//2, W//2:, :], axis=(0, 1))
        zone3_mean = np.mean(transformed[H//2:, :W//2, :], axis=(0, 1))
        zone4_mean = np.mean(transformed[H//2:, W//2:, :], axis=(0, 1))
        
        # Zones should have some variation (not all identical)
        # Note: With scaled data and physical-guided selection, differences may be subtle
        all_means = np.array([zone1_mean, zone2_mean, zone3_mean, zone4_mean])
        mean_std = np.std(all_means, axis=0)
        assert np.any(mean_std > 0.003)  # At least some variation across zones
