"""
Unit tests for WinnerTakesAllClassifier class.
WinnerTakesAllClassifier类的单元测试。
"""

import pytest
import numpy as np
from hyperspectral_pseudo_label_generator.config import ProcessingConfig
from hyperspectral_pseudo_label_generator.classification.classifier import WinnerTakesAllClassifier


class TestWinnerTakesAllClassifier:
    """Test suite for WinnerTakesAllClassifier class."""
    
    def test_no_class_passes_threshold(self):
        """Test that pixels with no passing classes get label 0."""
        config = ProcessingConfig()
        classifier = WinnerTakesAllClassifier(config)
        
        # Create SID scores and masks where no class passes
        H, W, num_classes = 5, 5, 5
        sid_scores = np.random.rand(H, W, num_classes) * 10
        masks = np.zeros((H, W, num_classes), dtype=bool)
        
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        # All pixels should be labeled 0 (unlabeled)
        assert pseudo_labels.shape == (H, W)
        assert pseudo_labels.dtype == np.uint8
        assert np.all(pseudo_labels == 0)
    
    def test_single_class_passes_threshold(self):
        """Test that pixels with single passing class get that class label."""
        config = ProcessingConfig()
        classifier = WinnerTakesAllClassifier(config)
        
        # Create SID scores and masks
        H, W, num_classes = 3, 3, 5
        sid_scores = np.random.rand(H, W, num_classes) * 10
        masks = np.zeros((H, W, num_classes), dtype=bool)
        
        # Pixel (0, 0) passes only class 0 (ore) → should get label 1
        masks[0, 0, 0] = True
        
        # Pixel (1, 1) passes only class 3 (poor ore pegmatite) → should get label 2
        masks[1, 1, 3] = True
        
        # Pixel (2, 2) passes only class 4 (wall rock) → should get label 3
        masks[2, 2, 4] = True
        
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        assert pseudo_labels[0, 0] == 1  # Ore
        assert pseudo_labels[1, 1] == 2  # Poor Ore Pegmatite
        assert pseudo_labels[2, 2] == 3  # Wall Rock
    
    def test_class_to_label_mapping(self):
        """Test correct mapping from class indices to output labels."""
        config = ProcessingConfig()
        classifier = WinnerTakesAllClassifier(config)
        
        # Test ore classes (0, 1, 2) → label 1
        assert classifier._map_class_to_label(0) == 1
        assert classifier._map_class_to_label(1) == 1
        assert classifier._map_class_to_label(2) == 1
        
        # Test poor ore pegmatite (3) → label 2
        assert classifier._map_class_to_label(3) == 2
        
        # Test wall rock (4) → label 3
        assert classifier._map_class_to_label(4) == 3
    
    def test_invalid_class_id_raises_error(self):
        """Test that invalid class IDs raise ValueError."""
        config = ProcessingConfig()
        classifier = WinnerTakesAllClassifier(config)
        
        with pytest.raises(ValueError, match="Invalid class_id"):
            classifier._map_class_to_label(5)
        
        with pytest.raises(ValueError, match="Invalid class_id"):
            classifier._map_class_to_label(-1)
    
    def test_winner_takes_all_clear_winner(self):
        """Test winner-takes-all when there's a clear winner."""
        config = ProcessingConfig(ambiguity_threshold=0.1)
        classifier = WinnerTakesAllClassifier(config)
        
        # Create SID scores and masks
        H, W, num_classes = 2, 2, 5
        sid_scores = np.ones((H, W, num_classes)) * 10.0
        masks = np.zeros((H, W, num_classes), dtype=bool)
        
        # Pixel (0, 0): classes 0 and 1 pass, but class 0 has much lower SID
        masks[0, 0, 0] = True
        masks[0, 0, 1] = True
        sid_scores[0, 0, 0] = 1.0  # Clear winner
        sid_scores[0, 0, 1] = 5.0  # Much higher
        
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        # Should assign label 1 (ore) since class 0 is clear winner
        assert pseudo_labels[0, 0] == 1
    
    def test_ambiguity_detection(self):
        """Test that ambiguous pixels get label 0."""
        config = ProcessingConfig(ambiguity_threshold=0.5)
        classifier = WinnerTakesAllClassifier(config)
        
        # Create SID scores and masks
        H, W, num_classes = 2, 2, 5
        sid_scores = np.ones((H, W, num_classes)) * 10.0
        masks = np.zeros((H, W, num_classes), dtype=bool)
        
        # Pixel (0, 0): classes 0 and 1 pass with very similar SID scores
        masks[0, 0, 0] = True
        masks[0, 0, 1] = True
        sid_scores[0, 0, 0] = 2.0
        sid_scores[0, 0, 1] = 2.1  # Difference < ambiguity_threshold
        
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        # Should assign label 0 (ambiguous)
        assert pseudo_labels[0, 0] == 0
    
    def test_multiple_ore_classes_pass(self):
        """Test that multiple ore classes passing still results in ore label."""
        config = ProcessingConfig(ambiguity_threshold=0.1)
        classifier = WinnerTakesAllClassifier(config)
        
        # Create SID scores and masks
        H, W, num_classes = 2, 2, 5
        sid_scores = np.ones((H, W, num_classes)) * 10.0
        masks = np.zeros((H, W, num_classes), dtype=bool)
        
        # Pixel (0, 0): classes 0, 1, 2 all pass (all ore), class 0 wins
        masks[0, 0, 0] = True
        masks[0, 0, 1] = True
        masks[0, 0, 2] = True
        sid_scores[0, 0, 0] = 1.0  # Winner
        sid_scores[0, 0, 1] = 3.0
        sid_scores[0, 0, 2] = 4.0
        
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        # Should assign label 1 (ore)
        assert pseudo_labels[0, 0] == 1
    
    def test_output_dtype_uint8(self):
        """Test that output pseudo-labels have dtype uint8."""
        config = ProcessingConfig()
        classifier = WinnerTakesAllClassifier(config)
        
        H, W, num_classes = 5, 5, 5
        sid_scores = np.random.rand(H, W, num_classes) * 10
        masks = np.random.rand(H, W, num_classes) > 0.8
        
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        assert pseudo_labels.dtype == np.uint8
    
    def test_output_values_in_valid_range(self):
        """Test that output labels are only {0, 1, 2, 3}."""
        config = ProcessingConfig()
        classifier = WinnerTakesAllClassifier(config)
        
        H, W, num_classes = 10, 10, 5
        sid_scores = np.random.rand(H, W, num_classes) * 10
        masks = np.random.rand(H, W, num_classes) > 0.7
        
        pseudo_labels = classifier.classify(sid_scores, masks)
        
        unique_labels = np.unique(pseudo_labels)
        assert all(label in [0, 1, 2, 3] for label in unique_labels)
    
    def test_resolve_ambiguity_with_two_classes(self):
        """Test ambiguity resolution with exactly two passing classes."""
        config = ProcessingConfig(ambiguity_threshold=0.2)
        classifier = WinnerTakesAllClassifier(config)
        
        # Test clear winner
        pixel_scores = np.array([1.0, 3.0, 10.0, 10.0, 10.0])
        passing_classes = np.array([0, 1])
        
        label = classifier._resolve_ambiguity(pixel_scores, passing_classes)
        assert label == 1  # Class 0 (ore) wins → label 1
        
        # Test ambiguous case
        pixel_scores = np.array([2.0, 2.1, 10.0, 10.0, 10.0])
        passing_classes = np.array([0, 1])
        
        label = classifier._resolve_ambiguity(pixel_scores, passing_classes)
        assert label == 0  # Ambiguous
    
    def test_resolve_ambiguity_with_multiple_classes(self):
        """Test ambiguity resolution with more than two passing classes."""
        config = ProcessingConfig(ambiguity_threshold=0.3)
        classifier = WinnerTakesAllClassifier(config)
        
        # Test with 3 passing classes, clear winner
        pixel_scores = np.array([1.0, 5.0, 6.0, 10.0, 10.0])
        passing_classes = np.array([0, 1, 2])
        
        label = classifier._resolve_ambiguity(pixel_scores, passing_classes)
        assert label == 1  # Class 0 (ore) wins → label 1
        
        # Test with 3 passing classes, ambiguous (two lowest are close)
        pixel_scores = np.array([1.0, 1.2, 6.0, 10.0, 10.0])
        passing_classes = np.array([0, 1, 2])
        
        label = classifier._resolve_ambiguity(pixel_scores, passing_classes)
        assert label == 0  # Ambiguous
