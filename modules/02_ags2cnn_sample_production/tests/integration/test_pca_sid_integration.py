"""
Integration tests for PCA and SID modules.
PCA和SID模块的集成测试。

This module tests the integration between PhysicalGuidedPCA and SIDCalculator,
specifically verifying that scaled PCA components can be properly normalized
to probability distributions for SID calculation.

该模块测试PhysicalGuidedPCA和SIDCalculator之间的集成，
特别是验证缩放后的PCA成分可以正确归一化为概率分布用于SID计算。
"""

import numpy as np
import pytest

from hyperspectral_pseudo_label_generator.config import ProcessingConfig
from hyperspectral_pseudo_label_generator.pca.physical_guided_pca import PhysicalGuidedPCA
from hyperspectral_pseudo_label_generator.sid.sid_calculator import SIDCalculator


class TestPCASIDIntegration:
    """Integration tests for PCA and SID modules."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig(n_components=5)
        
        # Create synthetic wavelengths
        self.wavelengths = np.linspace(400, 2500, 297)
        
        # Create synthetic hyperspectral image
        np.random.seed(42)
        self.image = np.random.rand(50, 50, 297) * 0.5 + 0.3  # Values in [0.3, 0.8]
        
        # Create synthetic reference spectra
        self.reference_spectra = {
            i: np.random.rand(297) * 0.5 + 0.3 
            for i in range(5)
        }
    
    def test_pca_output_can_be_normalized_to_probability(self):
        """
        Test that PCA-transformed and scaled data can be normalized to 
        probability distributions.
        
        测试PCA转换和缩放后的数据可以归一化为概率分布。
        """
        # Initialize PCA
        pca = PhysicalGuidedPCA(self.config, self.wavelengths)
        
        # Transform image
        transformed_image = pca.fit_transform(self.image)
        
        # Verify shape
        H, W, K = transformed_image.shape
        assert K == self.config.n_components
        
        # Verify values are in [0, 1] (scaled)
        assert np.all(transformed_image >= 0)
        assert np.all(transformed_image <= 1)
        
        # Initialize SID calculator
        sid_calc = SIDCalculator(self.config)
        
        # Reshape to (H*W, K) for probability normalization
        pixels = transformed_image.reshape(-1, K)
        
        # Normalize to probability distributions
        prob_dist = sid_calc._to_probability(pixels)
        
        # Verify shape is preserved
        assert prob_dist.shape == pixels.shape
        
        # Verify each row sums to 1
        row_sums = prob_dist.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6, atol=1e-9)
        
        # Verify all values are positive
        assert np.all(prob_dist > 0)
    
    def test_reference_spectra_can_be_normalized_to_probability(self):
        """
        Test that PCA-transformed reference spectra can be normalized to 
        probability distributions.
        
        测试PCA转换后的参考光谱可以归一化为概率分布。
        """
        # Initialize PCA
        pca = PhysicalGuidedPCA(self.config, self.wavelengths)
        
        # Transform image first (required for PCA fitting)
        pca.fit_transform(self.image)
        
        # Transform reference spectra
        transformed_spectra = pca.transform_reference(self.reference_spectra)
        
        # Initialize SID calculator
        sid_calc = SIDCalculator(self.config)
        
        # Normalize each reference spectrum to probability distribution
        for class_id, spectrum in transformed_spectra.items():
            # Reshape to (1, K) for probability normalization
            spectrum_2d = spectrum.reshape(1, -1)
            
            # Normalize to probability distribution
            prob_dist = sid_calc._to_probability(spectrum_2d)
            
            # Verify shape
            assert prob_dist.shape == spectrum_2d.shape
            
            # Verify sums to 1
            assert np.isclose(prob_dist.sum(), 1.0, rtol=1e-6, atol=1e-9)
            
            # Verify all values are positive
            assert np.all(prob_dist > 0)
    
    def test_probability_normalization_preserves_pca_space_consistency(self):
        """
        Test that probability normalization maintains consistency between
        image pixels and reference spectra in PCA space.
        
        测试概率归一化保持影像像素和参考光谱在PCA空间中的一致性。
        """
        # Initialize PCA
        pca = PhysicalGuidedPCA(self.config, self.wavelengths)
        
        # Transform image
        transformed_image = pca.fit_transform(self.image)
        
        # Transform reference spectra
        transformed_spectra = pca.transform_reference(self.reference_spectra)
        
        # Initialize SID calculator
        sid_calc = SIDCalculator(self.config)
        
        # Normalize image pixels
        H, W, K = transformed_image.shape
        pixels = transformed_image.reshape(-1, K)
        pixels_prob = sid_calc._to_probability(pixels)
        
        # Normalize reference spectra
        spectra_prob = {}
        for class_id, spectrum in transformed_spectra.items():
            spectrum_2d = spectrum.reshape(1, -1)
            spectra_prob[class_id] = sid_calc._to_probability(spectrum_2d)[0]
        
        # Verify all probability distributions sum to 1
        pixel_sums = pixels_prob.sum(axis=1)
        np.testing.assert_allclose(pixel_sums, 1.0, rtol=1e-6, atol=1e-9)
        
        for class_id, prob in spectra_prob.items():
            assert np.isclose(prob.sum(), 1.0, rtol=1e-6, atol=1e-9)
        
        # Verify all values are positive (epsilon prevents zeros)
        assert np.all(pixels_prob > 0)
        for prob in spectra_prob.values():
            assert np.all(prob > 0)
    
    def test_probability_normalization_with_edge_case_pca_values(self):
        """
        Test probability normalization with edge case PCA values
        (all zeros, very small values, etc.).
        
        测试边缘情况PCA值的概率归一化（全零、非常小的值等）。
        """
        # Initialize SID calculator
        sid_calc = SIDCalculator(self.config)
        
        # Test case 1: All zeros (after scaling, some components might be zero)
        zeros = np.zeros((10, 5))
        prob_zeros = sid_calc._to_probability(zeros)
        
        # Should create uniform distribution due to epsilon
        row_sums = prob_zeros.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6, atol=1e-9)
        
        # Each value should be approximately 1/K
        K = zeros.shape[1]
        expected_uniform = 1.0 / K
        np.testing.assert_allclose(prob_zeros, expected_uniform, rtol=1e-6)
        
        # Test case 2: Very small values
        small_values = np.full((10, 5), 1e-12)
        prob_small = sid_calc._to_probability(small_values)
        
        row_sums = prob_small.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6, atol=1e-9)
        
        # Test case 3: Mixed zeros and non-zeros
        mixed = np.array([
            [0.0, 0.5, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        prob_mixed = sid_calc._to_probability(mixed)
        
        row_sums = prob_mixed.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6, atol=1e-9)
        
        # All values should be positive due to epsilon
        assert np.all(prob_mixed > 0)
    
    def test_requirement_5_5_integration(self):
        """
        Integration test validating Requirement 5.5:
        System SHALL normalize scaled values to form valid probability 
        distributions (sum to 1) for SID calculation.
        
        集成测试验证需求5.5：
        系统应归一化缩放值以形成有效的概率分布(总和为1)用于SID计算。
        """
        # Initialize modules
        pca = PhysicalGuidedPCA(self.config, self.wavelengths)
        sid_calc = SIDCalculator(self.config)
        
        # Step 1: Transform image with PCA (produces scaled values in [0, 1])
        transformed_image = pca.fit_transform(self.image)
        
        # Step 2: Transform reference spectra
        transformed_spectra = pca.transform_reference(self.reference_spectra)
        
        # Step 3: Normalize image pixels to probability distributions
        H, W, K = transformed_image.shape
        pixels = transformed_image.reshape(-1, K)
        pixels_prob = sid_calc._to_probability(pixels)
        
        # Step 4: Normalize reference spectra to probability distributions
        spectra_prob = {}
        for class_id, spectrum in transformed_spectra.items():
            spectrum_2d = spectrum.reshape(1, -1)
            spectra_prob[class_id] = sid_calc._to_probability(spectrum_2d)[0]
        
        # Requirement 5.5 Validation: All probability distributions sum to 1
        
        # Validate image pixels
        pixel_sums = pixels_prob.sum(axis=1)
        np.testing.assert_allclose(
            pixel_sums, 
            1.0, 
            rtol=1e-6, 
            atol=1e-9,
            err_msg="Image pixel probability distributions must sum to 1 (Requirement 5.5)"
        )
        
        # Validate reference spectra
        for class_id, prob in spectra_prob.items():
            assert np.isclose(
                prob.sum(), 
                1.0, 
                rtol=1e-6, 
                atol=1e-9
            ), f"Reference spectrum {class_id} probability distribution must sum to 1 (Requirement 5.5)"
        
        # Additional validation: epsilon prevents zeros
        assert np.all(pixels_prob > 0), "Epsilon should prevent zero probabilities in image pixels"
        for class_id, prob in spectra_prob.items():
            assert np.all(prob > 0), f"Epsilon should prevent zero probabilities in reference spectrum {class_id}"
        
        # Validate numerical stability (no NaN or Inf)
        assert not np.any(np.isnan(pixels_prob)), "Probability distributions should not contain NaN"
        assert not np.any(np.isinf(pixels_prob)), "Probability distributions should not contain Inf"
        
        for class_id, prob in spectra_prob.items():
            assert not np.any(np.isnan(prob)), f"Reference spectrum {class_id} should not contain NaN"
            assert not np.any(np.isinf(prob)), f"Reference spectrum {class_id} should not contain Inf"
    
    def test_full_pca_to_sid_pipeline(self):
        """
        Test complete pipeline from PCA transformation to SID calculation.
        测试从PCA转换到SID计算的完整流程。
        
        This integration test validates that:
        1. PCA transforms image and reference spectra consistently
        2. Scaled PCA output can be normalized to probability distributions
        3. SID can be calculated on the probability distributions
        4. Output has correct shape and properties
        
        此集成测试验证：
        1. PCA一致地转换影像和参考光谱
        2. 缩放的PCA输出可以归一化为概率分布
        3. 可以在概率分布上计算SID
        4. 输出具有正确的形状和属性
        """
        # Create synthetic hyperspectral image
        # 创建合成高光谱影像
        H, W, bands = 20, 20, 297
        np.random.seed(42)
        image = np.random.rand(H, W, bands) * 0.5 + 0.3  # Range [0.3, 0.8]
        
        # Create wavelengths with Al-OH absorption band
        # 创建包含Al-OH吸收波段的波长
        wavelengths = np.linspace(400, 2500, bands)
        
        # Create reference spectra (5 classes)
        # 创建参考光谱(5个类别)
        reference_spectra = {
            i: np.random.rand(bands) * 0.5 + 0.3
            for i in range(5)
        }
        
        # Step 1: PCA transformation
        # 步骤1: PCA转换
        config = ProcessingConfig(n_components=5)
        pca = PhysicalGuidedPCA(config, wavelengths)
        
        # Transform image
        # 转换影像
        image_pca = pca.fit_transform(image)
        assert image_pca.shape == (H, W, 5), "PCA should reduce to 5 components"
        
        # Transform reference spectra
        # 转换参考光谱
        reference_pca = pca.transform_reference(reference_spectra)
        assert len(reference_pca) == 5, "Should have 5 reference spectra"
        for class_id, spectrum in reference_pca.items():
            assert spectrum.shape == (5,), f"Reference spectrum {class_id} should have 5 components"
        
        # Step 2: Calculate SID
        # 步骤2: 计算SID
        sid_calculator = SIDCalculator(config)
        sid_scores = sid_calculator.calculate(image_pca, reference_pca)
        
        # Validate output
        # 验证输出
        assert sid_scores.shape == (H, W, 5), "SID scores should have shape (H, W, num_classes)"
        assert np.all(sid_scores >= 0), "SID scores must be non-negative"
        assert np.all(np.isfinite(sid_scores)), "SID scores must be finite"
        
        # Verify that each pixel has different SID scores for different classes
        # 验证每个像素对不同类别有不同的SID分数
        # (unless by chance they're identical, which is very unlikely)
        pixel_scores = sid_scores[10, 10, :]
        assert len(np.unique(pixel_scores)) > 1, \
            "Pixel should have different SID scores for different classes"
        
        # Verify that SID scores vary across pixels
        # 验证SID分数在像素间变化
        # (different pixels should have different spectral signatures)
        scores_pixel1 = sid_scores[5, 5, 0]
        scores_pixel2 = sid_scores[15, 15, 0]
        # Allow them to be equal by chance, but check that not all are identical
        all_scores_class0 = sid_scores[:, :, 0].flatten()
        assert len(np.unique(all_scores_class0)) > 1, \
            "Different pixels should have different SID scores"
    
    def test_pca_sid_pipeline_validates_requirements(self):
        """
        Test that validates the integration of Requirements 4.5, 5.3, 5.5, 6.1-6.5.
        测试验证需求4.5, 5.3, 5.5, 6.1-6.5的集成。
        
        This test ensures that:
        - PCA transformation is consistent (Req 4.5, 5.3)
        - Probability normalization works (Req 5.5)
        - SID calculation is correct (Req 6.1, 6.2)
        - SID output has correct shape (Req 6.4, 6.5)
        """
        # Create test data
        H, W, bands = 15, 15, 297
        image = np.random.rand(H, W, bands)
        wavelengths = np.linspace(400, 2500, bands)
        reference_spectra = {i: np.random.rand(bands) for i in range(5)}
        
        # Initialize modules
        config = ProcessingConfig(n_components=5)
        pca = PhysicalGuidedPCA(config, wavelengths)
        sid_calculator = SIDCalculator(config)
        
        # PCA transformation
        image_pca = pca.fit_transform(image)
        reference_pca = pca.transform_reference(reference_spectra)
        
        # Requirement 4.5, 5.3: Same transformation matrix and scaling
        # This is implicitly validated by the PCA module, but we can verify
        # that both image and reference are in the same PCA space
        assert image_pca.shape[2] == reference_pca[0].shape[0], \
            "Image and reference must be in same PCA space (Req 4.5, 5.3)"
        
        # Requirement 5.5: Probability normalization
        # Convert a sample pixel to probability
        sample_pixel = image_pca[5, 5, :].reshape(1, -1)
        prob_pixel = sid_calculator._to_probability(sample_pixel)
        assert np.isclose(prob_pixel.sum(), 1.0, rtol=1e-6), \
            "Probability distribution must sum to 1 (Req 5.5)"
        
        # Calculate SID
        sid_scores = sid_calculator.calculate(image_pca, reference_pca)
        
        # Requirement 6.4: SID computed against all reference spectra
        assert sid_scores.shape[2] == len(reference_spectra), \
            "SID must be computed against all reference spectra (Req 6.4)"
        
        # Requirement 6.5: Output shape (H, W, num_classes)
        assert sid_scores.shape == (H, W, 5), \
            "SID output must have shape (H, W, num_classes) (Req 6.5)"
        
        # Requirement 6.1: SID symmetry
        # Pick a pixel and verify SID symmetry with one reference
        pixel_prob = sid_calculator._to_probability(image_pca[7, 7, :].reshape(1, -1))
        ref_prob = sid_calculator._to_probability(reference_pca[0].reshape(1, -1))
        sid_pr = sid_calculator.calculate_sid(pixel_prob, ref_prob)
        sid_rp = sid_calculator.calculate_sid(ref_prob, pixel_prob)
        np.testing.assert_allclose(sid_pr, sid_rp, rtol=1e-6,
                                   err_msg="SID must be symmetric (Req 6.1)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
