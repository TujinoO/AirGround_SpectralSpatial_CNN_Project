"""
Integration tests for output module.
输出模块的集成测试。
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
import logging

from hyperspectral_pseudo_label_generator.output import (
    OutputSerializer,
    VisualizationGenerator,
    StatisticsLogger
)


class TestOutputModuleIntegration:
    """Integration tests for complete output workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_complete_output_workflow(self, caplog):
        """Test complete output workflow: serialize, visualize, log."""
        # Create a realistic pseudo-label map
        np.random.seed(42)
        H, W = 50, 50
        pseudo_labels = np.random.randint(0, 4, size=(H, W), dtype=np.uint8)
        
        # Ensure we have some of each label
        pseudo_labels[0:10, 0:10] = 0  # Unlabeled
        pseudo_labels[10:20, 0:10] = 1  # Ore
        pseudo_labels[20:30, 0:10] = 2  # Wall Rock
        pseudo_labels[30:40, 0:10] = 3  # Barren Pegmatite
        
        # 1. Serialize
        output_path = os.path.join(self.test_dir, "pseudo_label_map.npy")
        OutputSerializer.save(pseudo_labels, output_path)
        
        # 2. Visualize
        VisualizationGenerator.generate(pseudo_labels, self.test_dir)
        
        # 3. Log statistics
        with caplog.at_level(logging.INFO):
            StatisticsLogger.log_summary(pseudo_labels)
        
        # Verify all outputs exist
        assert os.path.exists(output_path)
        assert os.path.exists(os.path.join(self.test_dir, 'pseudo_label_visualization.png'))
        
        # Verify serialization round-trip
        loaded = np.load(output_path)
        np.testing.assert_array_equal(loaded, pseudo_labels)
        
        # Verify logging occurred
        assert "Total pixels" in caplog.text
        assert "2500" in caplog.text  # 50x50 = 2500
    
    def test_output_with_nested_directories(self):
        """Test output workflow with nested directory structure."""
        pseudo_labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        
        # Create nested paths
        nested_dir = os.path.join(self.test_dir, "experiment1", "run1", "output")
        
        # Serialize to nested path
        output_path = os.path.join(nested_dir, "labels.npy")
        OutputSerializer.save(pseudo_labels, output_path)
        
        # Visualize to nested path
        VisualizationGenerator.generate(pseudo_labels, nested_dir)
        
        # Verify both outputs exist
        assert os.path.exists(output_path)
        assert os.path.exists(os.path.join(nested_dir, 'pseudo_label_visualization.png'))
    
    def test_output_preserves_label_distribution(self):
        """Test that output preserves label distribution correctly."""
        # Create a map with known distribution
        pseudo_labels = np.zeros((100, 100), dtype=np.uint8)
        pseudo_labels[0:25, :] = 0  # 25% unlabeled
        pseudo_labels[25:50, :] = 1  # 25% ore
        pseudo_labels[50:75, :] = 2  # 25% wall rock
        pseudo_labels[75:100, :] = 3  # 25% barren pegmatite
        
        # Serialize and load
        output_path = os.path.join(self.test_dir, "labels.npy")
        OutputSerializer.save(pseudo_labels, output_path)
        loaded = np.load(output_path)
        
        # Verify distribution is preserved
        unique_orig, counts_orig = np.unique(pseudo_labels, return_counts=True)
        unique_loaded, counts_loaded = np.unique(loaded, return_counts=True)
        
        np.testing.assert_array_equal(unique_orig, unique_loaded)
        np.testing.assert_array_equal(counts_orig, counts_loaded)
    
    def test_output_handles_edge_case_all_same_label(self, caplog):
        """Test output handles edge case where all pixels have same label."""
        # All pixels are ore
        pseudo_labels = np.ones((20, 20), dtype=np.uint8)
        
        # Should handle without errors
        output_path = os.path.join(self.test_dir, "labels.npy")
        OutputSerializer.save(pseudo_labels, output_path)
        VisualizationGenerator.generate(pseudo_labels, self.test_dir)
        
        with caplog.at_level(logging.INFO):
            StatisticsLogger.log_summary(pseudo_labels)
        
        # Verify outputs exist
        assert os.path.exists(output_path)
        assert os.path.exists(os.path.join(self.test_dir, 'pseudo_label_visualization.png'))
        
        # Verify logging shows 100% for one label
        assert "100.00%" in caplog.text
    
    def test_output_handles_mostly_unlabeled(self, caplog):
        """Test output handles case with mostly unlabeled pixels."""
        # 99.9% unlabeled, 0.1% labeled (below 1% threshold)
        pseudo_labels = np.zeros((100, 100), dtype=np.uint8)
        pseudo_labels[0:1, 0:10] = 1  # Only 10 labeled pixels out of 10000
        
        # Should handle without errors
        output_path = os.path.join(self.test_dir, "labels.npy")
        OutputSerializer.save(pseudo_labels, output_path)
        VisualizationGenerator.generate(pseudo_labels, self.test_dir)
        
        with caplog.at_level(logging.WARNING):
            StatisticsLogger.log_summary(pseudo_labels)
        
        # Should warn about low coverage
        assert "Low label coverage" in caplog.text
        
        # Verify outputs still created
        assert os.path.exists(output_path)
        assert os.path.exists(os.path.join(self.test_dir, 'pseudo_label_visualization.png'))
