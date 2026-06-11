"""
Unit tests for SID calculator module.
SID计算器模块的单元测试。
"""

import numpy as np
import pytest

from hyperspectral_pseudo_label_generator.config import ProcessingConfig
from hyperspectral_pseudo_label_generator.sid.sid_calculator import SIDCalculator


class TestSIDCalculator:
    """Test suite for SIDCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig()
        self.calculator = SIDCalculator(self.config)
    
    def test_initialization(self):
        """Test SIDCalculator initialization."""
        assert self.calculator.config == self.config
    
    def test_to_probability_basic(self):
        """Test basic probability distribution normalization."""
        # Create simple test data
        data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.5, 0.5, 1.0]
        ])
        
        # Convert to probability
        prob = self.calculator._to_probability(data)
        
        # Check shape is preserved
        assert prob.shape == data.shape
        
        # Check each row sums to 1
        row_sums = prob.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6, atol=1e-9)
        
        # Check all values are positive
        assert np.all(prob > 0)
    
    def test_to_probability_with_zeros(self):
        """Test probability normalization with zero values."""
        # Create data with zeros
        data = np.array([
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ])
        
        # Convert to probability
        prob = self.calculator._to_probability(data)
        
        # Check shape is preserved
        assert prob.shape == data.shape
        
        # Check each row sums to 1
        row_sums = prob.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6, atol=1e-9)
        
        # Check all values are positive (epsilon prevents zeros)
        assert np.all(prob > 0)
    
    def test_to_probability_all_zeros(self):
        """Test probability normalization when all values in a row are zero."""
        # Create data with all zeros in one row
        data = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0]
        ])
        
        # Convert to probability
        prob = self.calculator._to_probability(data)
        
        # Check shape is preserved
        assert prob.shape == data.shape
        
        # Check each row sums to 1
        row_sums = prob.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6, atol=1e-9)
        
        # Check all values are positive
        assert np.all(prob > 0)
        
        # For all-zero row, epsilon makes all values equal
        # So after normalization, each should be 1/K where K is number of features
        K = data.shape[1]
        expected_uniform = 1.0 / K
        np.testing.assert_allclose(prob[0, :], expected_uniform, rtol=1e-6)
    
    def test_to_probability_single_sample(self):
        """Test probability normalization with single sample."""
        # Create single sample
        data = np.array([[0.2, 0.3, 0.5]])
        
        # Convert to probability
        prob = self.calculator._to_probability(data)
        
        # Check shape is preserved
        assert prob.shape == data.shape
        
        # Check sums to 1
        assert np.isclose(prob.sum(), 1.0, rtol=1e-6, atol=1e-9)
        
        # Check all values are positive
        assert np.all(prob > 0)
    
    def test_to_probability_large_values(self):
        """Test probability normalization with large values."""
        # Create data with large values
        data = np.array([
            [1000.0, 2000.0, 3000.0],
            [500.0, 1500.0, 2500.0]
        ])
        
        # Convert to probability
        prob = self.calculator._to_probability(data)
        
        # Check shape is preserved
        assert prob.shape == data.shape
        
        # Check each row sums to 1
        row_sums = prob.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6, atol=1e-9)
        
        # Check all values are positive
        assert np.all(prob > 0)
    
    def test_to_probability_small_values(self):
        """Test probability normalization with very small values."""
        # Create data with very small values
        data = np.array([
            [1e-8, 2e-8, 3e-8],
            [1e-9, 1e-9, 1e-9]
        ])
        
        # Convert to probability
        prob = self.calculator._to_probability(data)
        
        # Check shape is preserved
        assert prob.shape == data.shape
        
        # Check each row sums to 1
        row_sums = prob.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6, atol=1e-9)
        
        # Check all values are positive
        assert np.all(prob > 0)
    
    def test_to_probability_epsilon_effect(self):
        """Test that epsilon prevents zero probabilities."""
        # Create data with zeros
        data = np.array([[0.0, 1.0, 0.0]])
        
        # Convert to probability
        prob = self.calculator._to_probability(data)
        
        # Check that even zero values become non-zero after adding epsilon
        assert np.all(prob > 0)
        
        # The zero values should have probability proportional to epsilon
        # The non-zero value should dominate but not be 1.0
        assert prob[0, 1] < 1.0  # Middle value should not be exactly 1
        assert prob[0, 0] > 0  # First value should be positive
        assert prob[0, 2] > 0  # Last value should be positive
    
    def test_to_probability_preserves_relative_magnitudes(self):
        """Test that normalization preserves relative magnitudes."""
        # Create data where one value is twice another
        data = np.array([[1.0, 2.0, 1.0]])
        
        # Convert to probability
        prob = self.calculator._to_probability(data)
        
        # The middle value should be approximately twice the others
        # (approximately because epsilon affects small values more)
        ratio = prob[0, 1] / prob[0, 0]
        assert ratio > 1.5  # Should be close to 2, but epsilon affects it
    
    def test_to_probability_2d_array(self):
        """Test probability normalization with 2D array (multiple samples)."""
        # Create multiple samples
        n_samples = 100
        n_features = 5
        np.random.seed(42)
        data = np.random.rand(n_samples, n_features)
        
        # Convert to probability
        prob = self.calculator._to_probability(data)
        
        # Check shape is preserved
        assert prob.shape == data.shape
        
        # Check each row sums to 1
        row_sums = prob.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6, atol=1e-9)
        
        # Check all values are positive
        assert np.all(prob > 0)
        
        # Check all values are less than or equal to 1
        assert np.all(prob <= 1.0)
    
    def test_to_probability_validates_requirement_5_5(self):
        """
        Test that validates Requirement 5.5:
        System SHALL normalize scaled values to form valid probability 
        distributions (sum to 1) for SID calculation.
        
        验证需求5.5：
        系统应归一化缩放值以形成有效的概率分布(总和为1)用于SID计算。
        """
        # Create scaled PCA component values (typical range [0, 1])
        scaled_data = np.array([
            [0.1, 0.3, 0.5, 0.2, 0.4],
            [0.0, 0.0, 1.0, 0.5, 0.3],
            [0.8, 0.2, 0.1, 0.9, 0.6]
        ])
        
        # Convert to probability distribution
        prob_dist = self.calculator._to_probability(scaled_data)
        
        # Requirement 5.5: Each pixel's feature vector should sum to 1
        row_sums = prob_dist.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 
            1.0, 
            rtol=1e-6, 
            atol=1e-9,
            err_msg="Probability distributions must sum to 1 (Requirement 5.5)"
        )
        
        # Additional validation: all values should be in [0, 1]
        assert np.all(prob_dist >= 0), "Probabilities must be non-negative"
        assert np.all(prob_dist <= 1), "Probabilities must not exceed 1"
        
        # Epsilon should prevent exact zeros
        assert np.all(prob_dist > 0), "Epsilon should prevent zero probabilities"
    
    def test_kl_divergence_identical_distributions(self):
        """Test KL divergence between identical distributions is zero."""
        # Create identical distributions
        p = np.array([[0.25, 0.25, 0.25, 0.25]])
        q = np.array([[0.25, 0.25, 0.25, 0.25]])
        
        # Calculate KL divergence
        kl = self.calculator._kl_divergence(p, q)
        
        # Should be very close to zero
        np.testing.assert_allclose(kl, 0.0, atol=1e-6)
    
    def test_kl_divergence_different_distributions(self):
        """Test KL divergence between different distributions is positive."""
        # Create different distributions
        p = np.array([[0.5, 0.3, 0.2]])
        q = np.array([[0.2, 0.3, 0.5]])
        
        # Calculate KL divergence
        kl = self.calculator._kl_divergence(p, q)
        
        # Should be positive
        assert kl[0] > 0
    
    def test_kl_divergence_multiple_samples(self):
        """Test KL divergence with multiple samples."""
        # Create multiple samples
        p = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.3, 0.2, 0.5]
        ])
        q = np.array([[0.33, 0.33, 0.34]])
        
        # Calculate KL divergence
        kl = self.calculator._kl_divergence(p, q)
        
        # Check shape
        assert kl.shape == (3,)
        
        # All should be non-negative
        assert np.all(kl >= 0)
    
    def test_kl_divergence_with_epsilon(self):
        """Test that epsilon prevents numerical issues."""
        # Create distributions with zeros
        p = np.array([[0.0, 0.5, 0.5]])
        q = np.array([[0.5, 0.25, 0.25]])
        
        # Calculate KL divergence (should not raise error due to epsilon)
        kl = self.calculator._kl_divergence(p, q)
        
        # Should be finite
        assert np.isfinite(kl[0])
        
        # Should be non-negative
        assert kl[0] >= 0
    
    def test_kl_divergence_asymmetry(self):
        """Test that KL divergence is asymmetric: D_KL(p||q) != D_KL(q||p)."""
        # Create different distributions that are clearly asymmetric
        # p is concentrated on first element, q is uniform
        p = np.array([[0.8, 0.1, 0.1]])
        q = np.array([[0.33, 0.33, 0.34]])
        
        # Calculate both directions
        kl_pq = self.calculator._kl_divergence(p, q)
        kl_qp = self.calculator._kl_divergence(q, p)
        
        # Should be different (asymmetric)
        # Note: They should both be positive but different values
        assert kl_pq[0] > 0
        assert kl_qp[0] > 0
        # KL divergence is generally asymmetric - use stricter tolerance
        assert not np.isclose(kl_pq[0], kl_qp[0], rtol=0.05)
    
    def test_kl_divergence_validates_requirement_6_2(self):
        """
        Test that validates Requirement 6.2:
        System SHALL implement KL_Divergence as: D_KL(p||q) = Σ p(i) * log(p(i)/q(i))
        
        验证需求6.2：
        系统应实现KL散度为: D_KL(p||q) = Σ p(i) * log(p(i)/q(i))
        """
        # Create simple probability distributions
        p = np.array([[0.5, 0.3, 0.2]])
        q = np.array([[0.2, 0.3, 0.5]])
        
        # Calculate KL divergence using the method
        kl_calculated = self.calculator._kl_divergence(p, q)
        
        # Calculate manually using the formula
        # D_KL(p||q) = Σ p(i) * log(p(i)/q(i))
        epsilon = self.config.epsilon
        p_safe = p + epsilon
        q_safe = q + epsilon
        kl_manual = np.sum(p_safe * np.log(p_safe / q_safe))
        
        # Should match
        np.testing.assert_allclose(
            kl_calculated[0],
            kl_manual,
            rtol=1e-6,
            err_msg="KL divergence must follow formula D_KL(p||q) = Σ p(i) * log(p(i)/q(i)) (Requirement 6.2)"
        )
    
    def test_kl_divergence_validates_requirement_6_3(self):
        """
        Test that validates Requirement 6.3:
        System SHALL add epsilon (1e-10) to prevent division by zero.
        
        验证需求6.3：
        系统应添加epsilon (1e-10) 以防止除以零。
        """
        # Create distributions with zeros that would cause division by zero
        p = np.array([[0.0, 0.5, 0.5]])
        q = np.array([[0.0, 0.5, 0.5]])
        
        # This should not raise an error due to epsilon
        try:
            kl = self.calculator._kl_divergence(p, q)
            # Should produce finite result
            assert np.isfinite(kl[0]), "KL divergence should be finite with epsilon"
            # Should be non-negative
            assert kl[0] >= 0, "KL divergence should be non-negative"
        except (ZeroDivisionError, RuntimeWarning):
            pytest.fail("KL divergence raised error despite epsilon (Requirement 6.3)")
    
    def test_kl_divergence_non_negative(self):
        """Test that KL divergence is always non-negative."""
        # Create random probability distributions
        np.random.seed(42)
        n_samples = 50
        n_features = 5
        
        # Generate random distributions
        p_raw = np.random.rand(n_samples, n_features)
        q_raw = np.random.rand(1, n_features)
        
        # Normalize to probability distributions
        p = p_raw / p_raw.sum(axis=1, keepdims=True)
        q = q_raw / q_raw.sum(axis=1, keepdims=True)
        
        # Calculate KL divergence
        kl = self.calculator._kl_divergence(p, q)
        
        # All values should be non-negative
        assert np.all(kl >= -1e-6), "KL divergence must be non-negative"
    
    def test_kl_divergence_broadcasting(self):
        """Test KL divergence with broadcasting (single q for multiple p)."""
        # Create multiple p distributions
        p = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.3, 0.2, 0.5]
        ])
        
        # Single q distribution (will be broadcast)
        q = np.array([[0.33, 0.33, 0.34]])
        
        # Calculate KL divergence
        kl = self.calculator._kl_divergence(p, q)
        
        # Check shape
        assert kl.shape == (3,)
        
        # All should be non-negative
        assert np.all(kl >= 0)
        
        # Each should be different (different p values)
        assert not np.allclose(kl[0], kl[1])
        assert not np.allclose(kl[1], kl[2])
    
    def test_calculate_sid_identical_distributions(self):
        """Test SID between identical distributions is zero."""
        # Create identical distributions
        p = np.array([[0.25, 0.25, 0.25, 0.25]])
        q = np.array([[0.25, 0.25, 0.25, 0.25]])
        
        # Calculate SID
        sid = self.calculator.calculate_sid(p, q)
        
        # Should be very close to zero
        np.testing.assert_allclose(sid, 0.0, atol=1e-6)
    
    def test_calculate_sid_different_distributions(self):
        """Test SID between different distributions is positive."""
        # Create different distributions
        p = np.array([[0.5, 0.3, 0.2]])
        q = np.array([[0.2, 0.3, 0.5]])
        
        # Calculate SID
        sid = self.calculator.calculate_sid(p, q)
        
        # Should be positive
        assert sid[0] > 0
    
    def test_calculate_sid_symmetry(self):
        """Test that SID is symmetric: SID(p, q) = SID(q, p)."""
        # Create different distributions
        p = np.array([[0.7, 0.2, 0.1]])
        q = np.array([[0.1, 0.2, 0.7]])
        
        # Calculate both directions
        sid_pq = self.calculator.calculate_sid(p, q)
        sid_qp = self.calculator.calculate_sid(q, p)
        
        # Should be equal (symmetric)
        np.testing.assert_allclose(sid_pq, sid_qp, rtol=1e-6)
    
    def test_calculate_sid_multiple_samples(self):
        """Test SID calculation with multiple samples."""
        # Create multiple samples
        p = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.3, 0.2, 0.5]
        ])
        q = np.array([[0.33, 0.33, 0.34]])
        
        # Calculate SID
        sid = self.calculator.calculate_sid(p, q)
        
        # Check shape
        assert sid.shape == (3,)
        
        # All should be non-negative
        assert np.all(sid >= 0)
    
    def test_calculate_sid_validates_requirement_6_1(self):
        """
        Test that validates Requirement 6.1:
        System SHALL implement SID as: SID(p,q) = D_KL(p||q) + D_KL(q||p)
        
        验证需求6.1：
        系统应实现SID为: SID(p,q) = D_KL(p||q) + D_KL(q||p)
        """
        # Create probability distributions
        p = np.array([[0.5, 0.3, 0.2]])
        q = np.array([[0.2, 0.3, 0.5]])
        
        # Calculate SID using the method
        sid_calculated = self.calculator.calculate_sid(p, q)
        
        # Calculate manually using the formula
        kl_pq = self.calculator._kl_divergence(p, q)
        kl_qp = self.calculator._kl_divergence(q, p)
        sid_manual = kl_pq + kl_qp
        
        # Should match
        np.testing.assert_allclose(
            sid_calculated,
            sid_manual,
            rtol=1e-6,
            err_msg="SID must follow formula SID(p,q) = D_KL(p||q) + D_KL(q||p) (Requirement 6.1)"
        )
    
    def test_calculate_sid_symmetry_validates_requirement_6_1(self):
        """
        Test that validates symmetry property from Requirement 6.1:
        SID(p,q) = SID(q,p)
        
        验证需求6.1的对称性属性：
        SID(p,q) = SID(q,p)
        """
        # Create different distributions
        p = np.array([[0.6, 0.3, 0.1]])
        q = np.array([[0.1, 0.4, 0.5]])
        
        # Calculate both directions
        sid_pq = self.calculator.calculate_sid(p, q)
        sid_qp = self.calculator.calculate_sid(q, p)
        
        # Should be equal (symmetric)
        np.testing.assert_allclose(
            sid_pq,
            sid_qp,
            rtol=1e-6,
            err_msg="SID must be symmetric: SID(p,q) = SID(q,p) (Requirement 6.1)"
        )
    
    def test_calculate_sid_non_negative(self):
        """Test that SID is always non-negative."""
        # Create random probability distributions
        np.random.seed(42)
        n_samples = 50
        n_features = 5
        
        # Generate random distributions
        p_raw = np.random.rand(n_samples, n_features)
        q_raw = np.random.rand(1, n_features)
        
        # Normalize to probability distributions
        p = p_raw / p_raw.sum(axis=1, keepdims=True)
        q = q_raw / q_raw.sum(axis=1, keepdims=True)
        
        # Calculate SID
        sid = self.calculator.calculate_sid(p, q)
        
        # All values should be non-negative
        assert np.all(sid >= -1e-6), "SID must be non-negative"
    
    def test_calculate_sid_with_zeros(self):
        """Test SID calculation with zero values (epsilon should handle)."""
        # Create distributions with zeros
        p = np.array([[0.0, 0.5, 0.5]])
        q = np.array([[0.5, 0.25, 0.25]])
        
        # Calculate SID (should not raise error due to epsilon)
        sid = self.calculator.calculate_sid(p, q)
        
        # Should be finite
        assert np.isfinite(sid[0])
        
        # Should be non-negative
        assert sid[0] >= 0
    
    def test_calculate_sid_lower_means_more_similar(self):
        """Test that lower SID scores indicate higher similarity."""
        # Create reference distribution
        q = np.array([[0.5, 0.3, 0.2]])
        
        # Create two distributions: one similar to q, one different
        p_similar = np.array([[0.52, 0.28, 0.20]])  # Very similar to q
        p_different = np.array([[0.1, 0.1, 0.8]])   # Very different from q
        
        # Calculate SID
        sid_similar = self.calculator.calculate_sid(p_similar, q)
        sid_different = self.calculator.calculate_sid(p_different, q)
        
        # Similar distribution should have lower SID
        assert sid_similar[0] < sid_different[0]
    
    def test_calculate_chunk_basic(self):
        """Test basic chunk SID calculation."""
        # Create small chunk
        chunk = np.array([
            [[0.2, 0.3, 0.5], [0.3, 0.4, 0.3]],
            [[0.5, 0.3, 0.2], [0.4, 0.3, 0.3]]
        ])  # Shape: (2, 2, 3)
        
        # Create reference spectra
        reference_spectra = {
            0: np.array([0.33, 0.33, 0.34]),
            1: np.array([0.5, 0.3, 0.2])
        }
        
        # Calculate SID for chunk
        sid_chunk = self.calculator._calculate_chunk(chunk, reference_spectra)
        
        # Check shape
        assert sid_chunk.shape == (2, 2, 2)
        
        # All values should be non-negative
        assert np.all(sid_chunk >= 0)
    
    def test_calculate_full_image_small(self):
        """Test full image SID calculation with small image."""
        # Create small test image
        np.random.seed(42)
        H, W, K = 10, 10, 5
        image = np.random.rand(H, W, K)
        
        # Normalize to [0, 1] range (simulating scaled PCA output)
        image = image / image.max()
        
        # Create reference spectra
        reference_spectra = {
            0: np.random.rand(K),
            1: np.random.rand(K),
            2: np.random.rand(K)
        }
        
        # Normalize reference spectra
        for class_id in reference_spectra:
            reference_spectra[class_id] = reference_spectra[class_id] / reference_spectra[class_id].max()
        
        # Calculate SID
        sid_scores = self.calculator.calculate(image, reference_spectra)
        
        # Check shape
        assert sid_scores.shape == (H, W, 3)
        
        # All values should be non-negative
        assert np.all(sid_scores >= 0)
        
        # All values should be finite
        assert np.all(np.isfinite(sid_scores))
    
    def test_calculate_validates_requirement_6_4(self):
        """
        Test that validates Requirement 6.4:
        System SHALL compute SID scores against all reference spectra in GSRSL.
        
        验证需求6.4：
        系统应计算该像素与GSRSL中所有参考光谱的SID分数。
        """
        # Create test image
        H, W, K = 5, 5, 3
        image = np.random.rand(H, W, K)
        
        # Create 5 reference spectra (as in GSRSL)
        reference_spectra = {
            0: np.random.rand(K),
            1: np.random.rand(K),
            2: np.random.rand(K),
            3: np.random.rand(K),
            4: np.random.rand(K)
        }
        
        # Calculate SID
        sid_scores = self.calculator.calculate(image, reference_spectra)
        
        # Requirement 6.4: Should compute SID against ALL reference spectra
        # Output should have num_classes dimension equal to number of reference spectra
        assert sid_scores.shape[2] == len(reference_spectra), \
            "SID scores must be computed against all reference spectra (Requirement 6.4)"
        
        # Each pixel should have SID score for each class
        for i in range(H):
            for j in range(W):
                for class_id in reference_spectra.keys():
                    assert np.isfinite(sid_scores[i, j, class_id]), \
                        f"SID score for pixel ({i},{j}) class {class_id} must be finite"
    
    def test_calculate_validates_requirement_6_5(self):
        """
        Test that validates Requirement 6.5:
        System SHALL return SID scores as 3D array with shape (H, W, num_classes).
        
        验证需求6.5：
        系统应返回形状为(H, W, num_classes)的3D数组的SID分数。
        """
        # Create test image
        H, W, K = 8, 12, 5
        image = np.random.rand(H, W, K)
        
        # Create reference spectra
        num_classes = 5
        reference_spectra = {i: np.random.rand(K) for i in range(num_classes)}
        
        # Calculate SID
        sid_scores = self.calculator.calculate(image, reference_spectra)
        
        # Requirement 6.5: Output shape must be (H, W, num_classes)
        expected_shape = (H, W, num_classes)
        assert sid_scores.shape == expected_shape, \
            f"SID scores must have shape (H, W, num_classes) = {expected_shape} (Requirement 6.5)"
        
        # Lower scores should indicate higher similarity (Requirement 6.5)
        # This is implicitly tested by the SID formula, but we can verify
        # that scores are non-negative (as required for divergence measures)
        assert np.all(sid_scores >= 0), \
            "SID scores must be non-negative (lower = more similar)"
    
    def test_calculate_chunked_processing(self):
        """Test that chunked processing produces correct results."""
        # Create test image larger than chunk size
        H, W, K = 25, 30, 5
        image = np.random.rand(H, W, K)
        
        # Create reference spectra
        reference_spectra = {
            0: np.random.rand(K),
            1: np.random.rand(K)
        }
        
        # Calculate SID with default chunk size
        sid_scores = self.calculator.calculate(image, reference_spectra)
        
        # Check shape
        assert sid_scores.shape == (H, W, 2)
        
        # All values should be non-negative
        assert np.all(sid_scores >= 0)
        
        # All values should be finite
        assert np.all(np.isfinite(sid_scores))
    
    def test_calculate_validates_requirement_15_1_15_2(self):
        """
        Test that validates Requirements 15.1 and 15.2:
        System SHALL use chunked processing for SID calculation to limit memory usage.
        System SHALL process SID calculations in spatial chunks.
        
        验证需求15.1和15.2：
        系统应使用分块处理进行SID计算以限制内存使用。
        系统应在空间块中处理SID计算。
        """
        # Create large image that would benefit from chunking
        H, W, K = 100, 100, 5
        image = np.random.rand(H, W, K)
        
        # Create reference spectra
        reference_spectra = {i: np.random.rand(K) for i in range(5)}
        
        # Set small chunk size to force chunking
        original_chunk_size = self.calculator.config.chunk_size
        self.calculator.config.chunk_size = 20
        
        try:
            # Calculate SID (should use chunked processing)
            sid_scores = self.calculator.calculate(image, reference_spectra)
            
            # Verify output shape is correct
            assert sid_scores.shape == (H, W, 5), \
                "Chunked processing must produce correct output shape (Requirements 15.1, 15.2)"
            
            # Verify all values are valid
            assert np.all(np.isfinite(sid_scores)), \
                "Chunked processing must produce finite values"
            
            # Verify non-negative
            assert np.all(sid_scores >= 0), \
                "Chunked processing must produce non-negative SID scores"
            
        finally:
            # Restore original chunk size
            self.calculator.config.chunk_size = original_chunk_size
    
    def test_calculate_different_chunk_sizes_same_result(self):
        """Test that different chunk sizes produce the same results."""
        # Create test image
        H, W, K = 30, 30, 5
        np.random.seed(42)
        image = np.random.rand(H, W, K)
        
        # Create reference spectra
        reference_spectra = {i: np.random.rand(K) for i in range(3)}
        
        # Calculate with different chunk sizes
        original_chunk_size = self.calculator.config.chunk_size
        
        try:
            # Large chunk size (process in fewer chunks)
            self.calculator.config.chunk_size = 20
            sid_large_chunks = self.calculator.calculate(image, reference_spectra)
            
            # Small chunk size (process in more chunks)
            self.calculator.config.chunk_size = 5
            sid_small_chunks = self.calculator.calculate(image, reference_spectra)
            
            # Results should be very similar (allowing for numerical differences)
            np.testing.assert_allclose(
                sid_large_chunks,
                sid_small_chunks,
                rtol=1e-5,
                atol=1e-8,
                err_msg="Different chunk sizes should produce same results"
            )
            
        finally:
            # Restore original chunk size
            self.calculator.config.chunk_size = original_chunk_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
