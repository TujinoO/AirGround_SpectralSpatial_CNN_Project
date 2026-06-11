"""
Integration tests for classification module.
分类模块的集成测试。
"""

import pytest
import numpy as np
from hyperspectral_pseudo_label_generator.config import ProcessingConfig
from hyperspectral_pseudo_label_generator.classification.thresholder import PercentileThresholder
from hyperspectral_pseudo_label_generator.classification.classifier import WinnerTakesAllClassifier


class TestClassificationIntegration:
    """Integration tests for the complete classification workflow."""
    
    def test_complete_classification_workflow(self):
        """Test the complete workflow from SID scores to pseudo-labels."""
        # Setup
        config = ProcessingConfig(
            ore_percentile=2.5,
            non_ore_percentile=5.0,
            ambiguity_threshold=0.1
        )
        thresholder = PercentileThresholder(config)
        classifier = WinnerTakesAllClassifier(config)
        
        # Create synthetic SID scores (H, W, num_classes)
        H, W, num_classes = 20, 20, 5
        sid_scores = np.random.rand(H, W, num_classes) * 10
        
        # Make some pixels clearly belong to specific classes
        # Ore pixels (class 0)
        sid_scores[0:5, 0:5, 0] = 0.1  # Very low SID for class 0
        sid_scores[0:5, 0:5, 1:] = 10.0  # High SID for other classes
        
        # Wall rock pixels (class 4)
        sid_scores[10:15, 10:15, 4] = 0.1
        sid_scores[10:15, 10:15, 0:4] = 10.0
        
        # Barren pegmatite pixels (class 3)
        sid_scores[15:20, 15:20, 3] = 0.1
        sid_scores[15:20, 15:20, [0, 1, 2, 4]] = 10.0
        
        # Step 1: Compute thresholds
        thresholds = thresholder.compute_thresholds(sid_scores)
        
        # Verify thresholds computed
        assert len(thresholds) == num_classes
        assert all(isinstance(v, (float, np.floating)) for v in thresholds.values())
        
        # Step 2: Apply thresholds
        masks = thresholder.apply_thresholds(sid_scores)
        
        # Verify masks shape and type
        assert masks.shape == (H, W, num_classes)
        assert masks.dtype == bool
        
        # Step 3: Classify pixels
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        # Verify output
        assert pseudo_labels.shape == (H, W)
        assert pseudo_labels.dtype == np.uint8
        
        # Verify labels are in valid range
        unique_labels = np.unique(pseudo_labels)
        assert all(label in [0, 1, 2, 3] for label in unique_labels)
        
        # Verify some ore pixels got label 1
        ore_region = pseudo_labels[0:5, 0:5]
        assert np.any(ore_region == 1)
        
        # Verify some wall rock pixels got label 3
        wall_rock_region = pseudo_labels[10:15, 10:15]
        assert np.any(wall_rock_region == 3)
        
        # Verify some poor ore pegmatite pixels got label 2
        barren_region = pseudo_labels[15:20, 15:20]
        assert np.any(barren_region == 2)
    
    def test_classification_with_realistic_sid_scores(self):
        """Test classification with more realistic SID score distributions."""
        config = ProcessingConfig(
            ore_percentile=2.5,
            non_ore_percentile=5.0,
            ambiguity_threshold=0.2
        )
        thresholder = PercentileThresholder(config)
        classifier = WinnerTakesAllClassifier(config)
        
        # Create realistic SID scores with different distributions per class
        H, W, num_classes = 50, 50, 5
        
        # Most pixels have high SID (dissimilar)
        sid_scores = np.random.exponential(scale=5.0, size=(H, W, num_classes))
        
        # Add some high-confidence ore pixels
        num_ore_pixels = 20
        ore_indices = np.random.choice(H * W, num_ore_pixels, replace=False)
        for idx in ore_indices:
            i, j = idx // W, idx % W
            # Low SID for one of the ore classes
            ore_class = np.random.choice([0, 1, 2])
            sid_scores[i, j, ore_class] = np.random.uniform(0.01, 0.1)
            # High SID for other classes
            for c in range(num_classes):
                if c != ore_class:
                    sid_scores[i, j, c] = np.random.uniform(5.0, 10.0)
        
        # Compute thresholds and classify
        thresholds = thresholder.compute_thresholds(sid_scores)
        masks = thresholder.apply_thresholds(sid_scores)
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        # Verify output structure
        assert pseudo_labels.shape == (H, W)
        assert pseudo_labels.dtype == np.uint8
        
        # Count labeled pixels
        labeled_pixels = np.sum(pseudo_labels > 0)
        total_pixels = H * W
        
        # With strict thresholds, we should have some labeled pixels
        # but not too many (the "better to miss than mislabel" philosophy)
        assert labeled_pixels > 0
        assert labeled_pixels < total_pixels * 0.2  # Less than 20% labeled
    
    def test_classification_handles_ambiguous_pixels(self):
        """Test that ambiguous pixels are correctly identified."""
        config = ProcessingConfig(
            ore_percentile=10.0,  # More lenient for testing
            non_ore_percentile=10.0,
            ambiguity_threshold=0.5
        )
        thresholder = PercentileThresholder(config)
        classifier = WinnerTakesAllClassifier(config)
        
        # Create SID scores with ambiguous pixels
        H, W, num_classes = 10, 10, 5
        sid_scores = np.ones((H, W, num_classes)) * 10.0
        
        # Create clear winner pixels
        sid_scores[0, 0, 0] = 1.0
        sid_scores[0, 0, 1:] = 5.0
        
        # Create ambiguous pixels (two classes very similar)
        sid_scores[5, 5, 0] = 2.0
        sid_scores[5, 5, 1] = 2.1  # Difference < ambiguity_threshold
        sid_scores[5, 5, 2:] = 10.0
        
        # Classify
        thresholds = thresholder.compute_thresholds(sid_scores)
        masks = thresholder.apply_thresholds(sid_scores)
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        # Clear winner should be labeled
        assert pseudo_labels[0, 0] == 1  # Ore
        
        # Ambiguous pixel should be unlabeled
        assert pseudo_labels[5, 5] == 0
    
    def test_classification_preserves_label_semantics(self):
        """Test that label semantics are preserved throughout classification."""
        config = ProcessingConfig()
        thresholder = PercentileThresholder(config)
        classifier = WinnerTakesAllClassifier(config)
        
        # Create controlled SID scores
        H, W, num_classes = 5, 5, 5
        sid_scores = np.ones((H, W, num_classes)) * 10.0
        
        # Pixel (0, 0): class 0 (ore type 1) wins
        sid_scores[0, 0, 0] = 0.1
        
        # Pixel (1, 1): class 1 (ore type 2) wins
        sid_scores[1, 1, 1] = 0.1
        
        # Pixel (2, 2): class 2 (ore type 3) wins
        sid_scores[2, 2, 2] = 0.1
        
        # Pixel (3, 3): class 3 (barren pegmatite) wins
        sid_scores[3, 3, 3] = 0.1
        
        # Pixel (4, 4): class 4 (wall rock) wins
        sid_scores[4, 4, 4] = 0.1
        
        # Classify
        thresholds = thresholder.compute_thresholds(sid_scores)
        masks = thresholder.apply_thresholds(sid_scores)
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        # Verify label mapping
        # All ore classes (0, 1, 2) should map to label 1
        assert pseudo_labels[0, 0] == 1
        assert pseudo_labels[1, 1] == 1
        assert pseudo_labels[2, 2] == 1
        
        # Poor ore pegmatite (class 3) should map to label 2
        assert pseudo_labels[3, 3] == 2
        
        # Wall rock (class 4) should map to label 3
        assert pseudo_labels[4, 4] == 3
    
    def test_classification_with_no_labeled_pixels(self):
        """Test classification when no pixels pass thresholds."""
        config = ProcessingConfig(
            ore_percentile=0.1,  # Extremely strict
            non_ore_percentile=0.1
        )
        thresholder = PercentileThresholder(config)
        classifier = WinnerTakesAllClassifier(config)
        
        # Create SID scores where all pixels are dissimilar
        H, W, num_classes = 10, 10, 5
        sid_scores = np.random.uniform(5.0, 10.0, size=(H, W, num_classes))
        
        # Classify
        thresholds = thresholder.compute_thresholds(sid_scores)
        masks = thresholder.apply_thresholds(sid_scores)
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        # Most or all pixels should be unlabeled
        unlabeled_count = np.sum(pseudo_labels == 0)
        assert unlabeled_count >= H * W * 0.9  # At least 90% unlabeled
    
    def test_classification_statistics(self):
        """Test that classification produces reasonable statistics."""
        config = ProcessingConfig(
            ore_percentile=5.0,
            non_ore_percentile=5.0,
            ambiguity_threshold=0.1
        )
        thresholder = PercentileThresholder(config)
        classifier = WinnerTakesAllClassifier(config)
        
        # Create diverse SID scores
        H, W, num_classes = 100, 100, 5
        sid_scores = np.random.rand(H, W, num_classes) * 10
        
        # Classify
        thresholds = thresholder.compute_thresholds(sid_scores)
        masks = thresholder.apply_thresholds(sid_scores)
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        # Compute statistics
        total_pixels = H * W
        unique, counts = np.unique(pseudo_labels, return_counts=True)
        label_counts = dict(zip(unique, counts))
        
        # Verify we have some labeled pixels
        labeled_pixels = total_pixels - label_counts.get(0, 0)
        assert labeled_pixels > 0
        
        # With 5% threshold, we expect roughly 5% * 5 classes = 25% labeled
        # But with ambiguity handling, it will be less
        coverage = labeled_pixels / total_pixels
        assert 0.01 < coverage < 0.5  # Between 1% and 50%
        
        # Verify all labels are valid
        assert all(label in [0, 1, 2, 3] for label in unique)
