"""
Unit tests for output module (serializer, visualizer, statistics).
输出模块的单元测试（序列化器、可视化器、统计）。
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
import logging
from pathlib import Path

from hyperspectral_pseudo_label_generator.output import (
    OutputSerializer,
    VisualizationGenerator,
    StatisticsLogger
)

try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class TestOutputSerializer:
    """Tests for OutputSerializer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_save_creates_npy_file(self):
        """Test that save creates a .npy file."""
        pseudo_labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        output_path = os.path.join(self.test_dir, "test_labels.npy")
        
        OutputSerializer.save(pseudo_labels, output_path)
        
        assert os.path.exists(output_path)
        assert output_path.endswith('.npy')
    
    def test_save_preserves_data(self):
        """Test that save preserves data correctly."""
        pseudo_labels = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.uint8)
        output_path = os.path.join(self.test_dir, "test_labels.npy")
        
        OutputSerializer.save(pseudo_labels, output_path)
        loaded = np.load(output_path)
        
        np.testing.assert_array_equal(loaded, pseudo_labels)
    
    def test_save_preserves_dtype(self):
        """Test that save preserves dtype (uint8)."""
        pseudo_labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        output_path = os.path.join(self.test_dir, "test_labels.npy")
        
        OutputSerializer.save(pseudo_labels, output_path)
        loaded = np.load(output_path)
        
        assert loaded.dtype == np.uint8
    
    def test_save_creates_directory_if_not_exists(self):
        """Test that save creates output directory if it doesn't exist."""
        nested_dir = os.path.join(self.test_dir, "nested", "path")
        output_path = os.path.join(nested_dir, "test_labels.npy")
        
        OutputSerializer.save(np.array([[0, 1]], dtype=np.uint8), output_path)
        
        assert os.path.exists(nested_dir)
        assert os.path.exists(output_path)
    
    def test_save_handles_large_arrays(self):
        """Test that save handles large arrays."""
        # Create a larger pseudo-label map
        pseudo_labels = np.random.randint(0, 4, size=(100, 100), dtype=np.uint8)
        output_path = os.path.join(self.test_dir, "large_labels.npy")
        
        OutputSerializer.save(pseudo_labels, output_path)
        loaded = np.load(output_path)
        
        np.testing.assert_array_equal(loaded, pseudo_labels)
        assert loaded.shape == (100, 100)

    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not installed")
    def test_save_with_reference_creates_geotiff_with_georef(self):
        pseudo_labels = np.random.randint(0, 4, size=(10, 12), dtype=np.uint8)

        reference_path = os.path.join(self.test_dir, "reference.tif")
        transform = from_bounds(100.0, 20.0, 112.0, 30.0, 12, 10)
        crs = "EPSG:4326"
        reference_data = np.random.rand(1, 10, 12).astype(np.float32)

        with rasterio.open(
            reference_path,
            "w",
            driver="GTiff",
            height=10,
            width=12,
            count=1,
            dtype=reference_data.dtype,
            transform=transform,
            crs=crs,
        ) as dst:
            dst.write(reference_data)

        output_path = os.path.join(self.test_dir, "labels.tif")
        OutputSerializer.save_with_reference(pseudo_labels, output_path, reference_image_path=reference_path)

        with rasterio.open(reference_path) as ref_ds, rasterio.open(output_path) as out_ds:
            assert out_ds.width == ref_ds.width
            assert out_ds.height == ref_ds.height
            assert out_ds.transform == ref_ds.transform
            assert out_ds.crs == ref_ds.crs
            assert out_ds.count == 1
            assert out_ds.dtypes[0] == "uint8"
            assert out_ds.nodata == 0
            np.testing.assert_array_equal(out_ds.read(1), pseudo_labels)


class TestVisualizationGenerator:
    """Tests for VisualizationGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_generate_creates_png_file(self):
        """Test that generate creates a PNG file."""
        pseudo_labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        
        VisualizationGenerator.generate(pseudo_labels, self.test_dir)
        
        output_path = os.path.join(self.test_dir, 'pseudo_label_visualization.png')
        assert os.path.exists(output_path)
        assert output_path.endswith('.png')
    
    def test_generate_creates_directory_if_not_exists(self):
        """Test that generate creates output directory if it doesn't exist."""
        nested_dir = os.path.join(self.test_dir, "nested", "viz")
        pseudo_labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        
        VisualizationGenerator.generate(pseudo_labels, nested_dir)
        
        assert os.path.exists(nested_dir)
        output_path = os.path.join(nested_dir, 'pseudo_label_visualization.png')
        assert os.path.exists(output_path)
    
    def test_generate_handles_all_labels(self):
        """Test that generate handles all label values {0, 1, 2, 3}."""
        pseudo_labels = np.array([
            [0, 1, 2, 3],
            [3, 2, 1, 0]
        ], dtype=np.uint8)
        
        # Should not raise an exception
        VisualizationGenerator.generate(pseudo_labels, self.test_dir)
        
        output_path = os.path.join(self.test_dir, 'pseudo_label_visualization.png')
        assert os.path.exists(output_path)
    
    def test_generate_handles_single_label(self):
        """Test that generate handles maps with only one label."""
        pseudo_labels = np.zeros((10, 10), dtype=np.uint8)
        
        # Should not raise an exception
        VisualizationGenerator.generate(pseudo_labels, self.test_dir)
        
        output_path = os.path.join(self.test_dir, 'pseudo_label_visualization.png')
        assert os.path.exists(output_path)
    
    def test_generate_handles_larger_arrays(self):
        """Test that generate handles larger arrays."""
        pseudo_labels = np.random.randint(0, 4, size=(50, 50), dtype=np.uint8)
        
        VisualizationGenerator.generate(pseudo_labels, self.test_dir)
        
        output_path = os.path.join(self.test_dir, 'pseudo_label_visualization.png')
        assert os.path.exists(output_path)

    def test_generate_with_rgb_overlay_creates_png_file(self):
        pseudo_labels = np.random.randint(0, 4, size=(20, 30), dtype=np.uint8)
        wavelengths = np.linspace(400.0, 2500.0, 297).astype(np.float32)
        image = np.random.rand(20, 30, 297).astype(np.float32)

        VisualizationGenerator.generate_with_rgb_overlay(
            pseudo_labels,
            self.test_dir,
            image=image,
            wavelengths=wavelengths,
        )

        output_path = os.path.join(self.test_dir, 'pseudo_label_overlay_rgb.png')
        assert os.path.exists(output_path)

    def test_labels_to_rgba_supports_wall_rock(self):
        pseudo_labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)

        overlay = VisualizationGenerator._labels_to_rgba(pseudo_labels, alpha=0.65)

        assert overlay.shape == (2, 2, 4)
        assert overlay[1, 1, 3] == pytest.approx(0.65)
        assert overlay[1, 1, 2] > 0.0

    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not installed")
    def test_get_plot_extent_and_labels_uses_lonlat_for_projected_reference(self):
        reference_path = os.path.join(self.test_dir, "reference_utm.tif")
        transform = from_bounds(688000.0, 4294000.0, 689200.0, 4295200.0, 12, 12)
        reference_data = np.random.rand(1, 12, 12).astype(np.float32)

        with rasterio.open(
            reference_path,
            "w",
            driver="GTiff",
            height=12,
            width=12,
            count=1,
            dtype=reference_data.dtype,
            transform=transform,
            crs="EPSG:32645",
        ) as dst:
            dst.write(reference_data)

        extent, xlabel, ylabel, is_geo = VisualizationGenerator._get_plot_extent_and_labels(
            (12, 12),
            reference_path,
        )

        expected_extent = rasterio.warp.transform_bounds(
            "EPSG:32645",
            "EPSG:4326",
            688000.0,
            4294000.0,
            689200.0,
            4295200.0,
            densify_pts=21,
        )
        assert extent == pytest.approx(expected_extent)
        assert xlabel == "经度"
        assert ylabel == "纬度"
        assert is_geo is True


class TestStatisticsLogger:
    """Tests for StatisticsLogger class."""
    
    def test_log_summary_logs_total_pixels(self, caplog):
        """Test that log_summary logs total pixel count."""
        pseudo_labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        
        with caplog.at_level(logging.INFO):
            StatisticsLogger.log_summary(pseudo_labels)
        
        assert "Total pixels" in caplog.text
        assert "4" in caplog.text
    
    def test_log_summary_logs_all_labels(self, caplog):
        """Test that log_summary logs counts for all labels."""
        pseudo_labels = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ], dtype=np.uint8)
        
        with caplog.at_level(logging.INFO):
            StatisticsLogger.log_summary(pseudo_labels)
        
        assert "Unlabeled/Ambiguous" in caplog.text
        assert "Rich Ore Pegmatite" in caplog.text
        assert "Poor Ore Pegmatite" in caplog.text
        assert "Wall Rock" in caplog.text
    
    def test_log_summary_calculates_percentages(self, caplog):
        """Test that log_summary calculates percentages correctly."""
        # 50% unlabeled, 50% ore
        pseudo_labels = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        
        with caplog.at_level(logging.INFO):
            StatisticsLogger.log_summary(pseudo_labels)
        
        assert "50.00%" in caplog.text
    
    def test_log_summary_warns_on_low_coverage(self, caplog):
        """Test that log_summary warns when coverage < 1%."""
        # 99.5% unlabeled, 0.5% labeled
        pseudo_labels = np.zeros((100, 100), dtype=np.uint8)
        pseudo_labels[0, 0] = 1  # Only 1 labeled pixel out of 10000
        
        with caplog.at_level(logging.WARNING):
            StatisticsLogger.log_summary(pseudo_labels)
        
        assert "Low label coverage" in caplog.text
    
    def test_log_summary_no_warning_on_good_coverage(self, caplog):
        """Test that log_summary doesn't warn when coverage >= 1%."""
        # 50% labeled
        pseudo_labels = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        
        with caplog.at_level(logging.WARNING):
            StatisticsLogger.log_summary(pseudo_labels)
        
        assert "Low label coverage" not in caplog.text
    
    def test_log_summary_handles_all_unlabeled(self, caplog):
        """Test that log_summary handles all pixels unlabeled."""
        pseudo_labels = np.zeros((10, 10), dtype=np.uint8)
        
        with caplog.at_level(logging.INFO):
            StatisticsLogger.log_summary(pseudo_labels)
        
        assert "Total pixels" in caplog.text
        assert "100" in caplog.text
    
    def test_log_summary_handles_no_unlabeled(self, caplog):
        """Test that log_summary handles no unlabeled pixels."""
        pseudo_labels = np.ones((10, 10), dtype=np.uint8)
        
        with caplog.at_level(logging.INFO):
            StatisticsLogger.log_summary(pseudo_labels)
        
        # Should not warn about low coverage
        with caplog.at_level(logging.WARNING):
            StatisticsLogger.log_summary(pseudo_labels)
        
        assert "Low label coverage" not in caplog.text
