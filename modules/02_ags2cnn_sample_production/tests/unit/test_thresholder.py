"""
Unit tests for PercentileThresholder class.
PercentileThresholder类的单元测试。
"""

import pytest
import numpy as np
from hyperspectral_pseudo_label_generator.config import ProcessingConfig
from hyperspectral_pseudo_label_generator.classification.thresholder import PercentileThresholder


class TestPercentileThresholder:
    """Test suite for PercentileThresholder class."""
    
    def test_compute_thresholds_ore_classes(self):
        """Test that ore classes use 2.5th percentile threshold."""
        config = ProcessingConfig(ore_percentile=2.5, non_ore_percentile=5.0)
        thresholder = PercentileThresholder(config)
        
        # Create synthetic SID scores
        H, W, num_classes = 10, 10, 5
        sid_scores = np.random.rand(H, W, num_classes) * 10
        
        thresholds = thresholder.compute_thresholds(sid_scores)
        
        # Verify thresholds computed for all classes
        assert len(thresholds) == num_classes
        
        # Verify ore classes (0, 1, 2) use 2.5th percentile
        for class_id in [0, 1, 2]:
            class_scores = sid_scores[:, :, class_id].flatten()
            expected_threshold = np.percentile(class_scores, 2.5)
            assert np.isclose(thresholds[class_id], expected_threshold)
    
    def test_compute_thresholds_non_ore_classes(self):
        """Test that non-ore classes use 5th percentile threshold."""
        config = ProcessingConfig(ore_percentile=2.5, non_ore_percentile=5.0)
        thresholder = PercentileThresholder(config)
        
        # Create synthetic SID scores
        H, W, num_classes = 10, 10, 5
        sid_scores = np.random.rand(H, W, num_classes) * 10
        
        thresholds = thresholder.compute_thresholds(sid_scores)
        
        # Verify non-ore classes (3, 4) use 5th percentile
        for class_id in [3, 4]:
            class_scores = sid_scores[:, :, class_id].flatten()
            expected_threshold = np.percentile(class_scores, 5.0)
            assert np.isclose(thresholds[class_id], expected_threshold)
    
    def test_apply_thresholds_creates_binary_masks(self):
        """Test that apply_thresholds creates correct binary masks."""
        config = ProcessingConfig()
        thresholder = PercentileThresholder(config)
        
        # Create synthetic SID scores
        H, W, num_classes = 5, 5, 5
        sid_scores = np.random.rand(H, W, num_classes) * 10
        
        # Compute thresholds
        thresholds = thresholder.compute_thresholds(sid_scores)
        
        # Apply thresholds
        masks = thresholder.apply_thresholds(sid_scores)
        
        # Verify output shape
        assert masks.shape == (H, W, num_classes)
        assert masks.dtype == bool
        
        # Verify masks are correct
        for class_id in range(num_classes):
            expected_mask = sid_scores[:, :, class_id] <= thresholds[class_id]
            assert np.array_equal(masks[:, :, class_id], expected_mask)
    
    def test_apply_thresholds_without_compute_raises_error(self):
        """Test that apply_thresholds raises error if thresholds not computed."""
        config = ProcessingConfig()
        thresholder = PercentileThresholder(config)
        
        # Create synthetic SID scores
        H, W, num_classes = 5, 5, 5
        sid_scores = np.random.rand(H, W, num_classes) * 10
        
        # Try to apply thresholds without computing them first
        with pytest.raises(RuntimeError, match="Must compute thresholds first"):
            thresholder.apply_thresholds(sid_scores)
    
    def test_thresholds_independent_per_class(self):
        """Test that thresholds are computed independently for each class."""
        config = ProcessingConfig(ore_percentile=2.5, non_ore_percentile=5.0)
        thresholder = PercentileThresholder(config)
        
        # Create SID scores with different distributions per class
        # Use seed for reproducibility
        np.random.seed(42)
        H, W, num_classes = 10, 10, 5
        sid_scores = np.zeros((H, W, num_classes))
        
        # Class 0: low values (mean ~0.5)
        sid_scores[:, :, 0] = np.random.rand(H, W) * 1
        # Class 1: medium values (mean ~4)
        sid_scores[:, :, 1] = np.random.rand(H, W) * 5 + 2
        # Class 2: high values (mean ~10)
        sid_scores[:, :, 2] = np.random.rand(H, W) * 10 + 5
        # Class 3: very low values (mean ~0.25)
        sid_scores[:, :, 3] = np.random.rand(H, W) * 0.5
        # Class 4: very high values (mean ~20)
        sid_scores[:, :, 4] = np.random.rand(H, W) * 20 + 10
        
        thresholds = thresholder.compute_thresholds(sid_scores)
        
        # Verify thresholds reflect the distributions
        # Class 3 should have lowest threshold (lowest values)
        assert thresholds[3] < thresholds[0]
        # Class 0 should have lower threshold than class 1
        assert thresholds[0] < thresholds[1]
        # Class 1 should have lower threshold than class 2
        assert thresholds[1] < thresholds[2]
        # Class 4 should have highest threshold (highest values)
        assert thresholds[4] > thresholds[2]
    
    def test_low_sid_scores_pass_threshold(self):
        """Test that pixels with low SID scores pass the threshold."""
        config = ProcessingConfig(ore_percentile=1.0)  # Very strict threshold
        thresholder = PercentileThresholder(config)
        
        # Create SID scores where we know which pixels should pass
        H, W, num_classes = 10, 10, 5
        sid_scores = np.ones((H, W, num_classes)) * 10.0
        
        # Set exactly one pixel to very low value (should pass threshold)
        sid_scores[0, 0, 0] = 0.1
        sid_scores[1, 1, 1] = 0.2
        
        thresholds = thresholder.compute_thresholds(sid_scores)
        masks = thresholder.apply_thresholds(sid_scores)
        
        # Verify low-scoring pixels pass
        assert masks[0, 0, 0] == True
        assert masks[1, 1, 1] == True
        
        # Verify most high-scoring pixels don't pass (at 1% threshold, only 1 pixel should pass)
        # Count how many pixels pass for class 0
        num_passing_class_0 = np.sum(masks[:, :, 0])
        # Should be very few (around 1% of 100 pixels = 1 pixel)
        assert num_passing_class_0 <= 2

    def test_compute_thresholds_3class_uses_poor_and_wall_percentiles(self):
        config = ProcessingConfig(
            ore_percentile=2.6,
            poor_percentile=9.0,
            wall_percentile=8.5,
            non_ore_percentile=6.5,
        )
        thresholder = PercentileThresholder(config)

        h, w = 12, 7
        sid_scores = np.random.rand(h, w, 3) * 5
        thresholds = thresholder.compute_thresholds(sid_scores)

        expected_rich = np.percentile(sid_scores[:, :, 0].flatten(), 2.6)
        expected_poor = np.percentile(sid_scores[:, :, 1].flatten(), 9.0)
        expected_wall = np.percentile(sid_scores[:, :, 2].flatten(), 8.5)
        assert np.isclose(thresholds[0], expected_rich)
        assert np.isclose(thresholds[1], expected_poor)
        assert np.isclose(thresholds[2], expected_wall)
