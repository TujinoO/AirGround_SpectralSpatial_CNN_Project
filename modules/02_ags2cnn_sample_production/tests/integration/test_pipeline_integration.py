"""
Integration tests for the complete pipeline.
完整流程的集成测试。
"""

import os
import tempfile
import shutil

import numpy as np
import pytest

from hyperspectral_pseudo_label_generator import ProcessingConfig, PseudoLabelPipeline


@pytest.fixture
def synthetic_data():
    """
    Create synthetic test data.
    创建合成测试数据。
    """
    # Create temporary directory
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    # Create synthetic image
    # 创建合成影像
    H, W, bands = 30, 30, 297
    image = np.random.rand(H, W, bands).astype(np.float32) * 0.5 + 0.2
    image_path = os.path.join(temp_dir, "test_image.npy")
    np.save(image_path, image)
    
    # Create synthetic reference spectra
    # 创建合成参考光谱
    reference_spectra = np.random.rand(5, bands).astype(np.float32) * 0.3 + 0.3
    reference_path = os.path.join(temp_dir, "test_reference.npy")
    np.save(reference_path, reference_spectra)
    
    # Create synthetic wavelengths
    # 创建合成波长
    wavelengths = np.linspace(400, 2500, bands, dtype=np.float32)
    metadata_path = os.path.join(temp_dir, "test_wavelengths.csv")
    np.savetxt(metadata_path, wavelengths, delimiter=',', fmt='%.2f')
    
    # Create output directory
    # 创建输出目录
    output_dir = os.path.join(temp_dir, "output")
    
    yield {
        'image_path': image_path,
        'reference_path': reference_path,
        'metadata_path': metadata_path,
        'output_dir': output_dir,
        'temp_dir': temp_dir
    }
    
    # Cleanup
    # 清理
    shutil.rmtree(temp_dir)


def test_pipeline_end_to_end(synthetic_data):
    """
    Test complete pipeline execution.
    测试完整流程执行。
    """
    # Create configuration
    # 创建配置
    config = ProcessingConfig(
        n_components=3,
        ore_percentile=5.0,
        non_ore_percentile=10.0,
        ambiguity_threshold=0.1,
        chunk_size=500,
        output_dir=synthetic_data['output_dir']
    )
    
    # Create and run pipeline
    # 创建并运行流程
    pipeline = PseudoLabelPipeline(config)
    
    pseudo_labels = pipeline.run(
        image_path=synthetic_data['image_path'],
        reference_path=synthetic_data['reference_path'],
        metadata_path=synthetic_data['metadata_path'],
        output_filename="test_output.npy"
    )
    
    # Verify output shape
    # 验证输出形状
    assert pseudo_labels.shape == (30, 30)
    assert pseudo_labels.dtype == np.uint8
    
    # Verify output values are in valid range
    # 验证输出值在有效范围内
    unique_labels = np.unique(pseudo_labels)
    assert all(label in [0, 1, 2, 3] for label in unique_labels)
    
    # Verify output files were created
    # 验证输出文件已创建
    output_file = os.path.join(synthetic_data['output_dir'], "test_output.npy")
    assert os.path.exists(output_file)
    
    viz_file = os.path.join(synthetic_data['output_dir'], "pseudo_label_visualization.png")
    assert os.path.exists(viz_file)

    overlay_file = os.path.join(synthetic_data['output_dir'], "pseudo_label_overlay_rgb.png")
    assert os.path.exists(overlay_file)
    
    # Verify saved file matches returned array
    # 验证保存的文件与返回的数组匹配
    loaded_labels = np.load(output_file)
    np.testing.assert_array_equal(loaded_labels, pseudo_labels)


def test_pipeline_with_custom_config(synthetic_data):
    """
    Test pipeline with custom configuration.
    使用自定义配置测试流程。
    """
    # Create custom configuration
    # 创建自定义配置
    config = ProcessingConfig(
        n_components=5,
        ore_percentile=2.5,
        non_ore_percentile=5.0,
        ambiguity_threshold=0.05,
        epsilon=1e-12,
        chunk_size=1000,
        aloh_min_wavelength=2100.0,
        aloh_max_wavelength=2300.0,
        output_dir=synthetic_data['output_dir']
    )
    
    # Run pipeline
    # 运行流程
    pipeline = PseudoLabelPipeline(config)
    
    pseudo_labels = pipeline.run(
        image_path=synthetic_data['image_path'],
        reference_path=synthetic_data['reference_path'],
        metadata_path=synthetic_data['metadata_path']
    )
    
    # Verify output
    # 验证输出
    assert pseudo_labels.shape == (30, 30)
    assert pseudo_labels.dtype == np.uint8


def test_pipeline_error_handling_invalid_image(synthetic_data):
    """
    Test pipeline error handling with invalid image.
    使用无效影像测试流程错误处理。
    """
    # Create invalid image (wrong number of bands)
    # 创建无效影像（错误的波段数）
    invalid_image = np.random.rand(30, 30, 100).astype(np.float32)
    invalid_path = os.path.join(synthetic_data['temp_dir'], "invalid_image.npy")
    np.save(invalid_path, invalid_image)
    
    # Create configuration
    # 创建配置
    config = ProcessingConfig(output_dir=synthetic_data['output_dir'])
    pipeline = PseudoLabelPipeline(config)
    
    # Should raise ValueError
    # 应该引发ValueError
    with pytest.raises(ValueError, match="297 bands"):
        pipeline.run(
            image_path=invalid_path,
            reference_path=synthetic_data['reference_path'],
            metadata_path=synthetic_data['metadata_path']
        )


def test_pipeline_error_handling_missing_file(synthetic_data):
    """
    Test pipeline error handling with missing file.
    使用缺失文件测试流程错误处理。
    """
    # Create configuration
    # 创建配置
    config = ProcessingConfig(output_dir=synthetic_data['output_dir'])
    pipeline = PseudoLabelPipeline(config)
    
    # Should raise IOError
    # 应该引发IOError
    with pytest.raises(IOError, match="File not found"):
        pipeline.run(
            image_path="nonexistent_file.npy",
            reference_path=synthetic_data['reference_path'],
            metadata_path=synthetic_data['metadata_path']
        )


def test_pipeline_statistics_logging(synthetic_data, caplog):
    """
    Test that pipeline logs statistics correctly.
    测试流程正确记录统计信息。
    """
    import logging
    
    # Create configuration
    # 创建配置
    config = ProcessingConfig(output_dir=synthetic_data['output_dir'])
    pipeline = PseudoLabelPipeline(config)
    
    # Run pipeline with logging
    # 运行流程并记录日志
    with caplog.at_level(logging.INFO):
        pipeline.run(
            image_path=synthetic_data['image_path'],
            reference_path=synthetic_data['reference_path'],
            metadata_path=synthetic_data['metadata_path']
        )
    
    # Verify key log messages are present
    # 验证关键日志消息存在
    log_text = caplog.text
    
    assert "Starting Pseudo-Label Generation Pipeline" in log_text
    assert "Loading and validating inputs" in log_text
    assert "Performing Physical-Guided MNF" in log_text
    assert "Calculating Spectral Information Divergence" in log_text
    assert "Applying percentile-based thresholding" in log_text
    assert "Classifying pixels with winner-takes-all" in log_text
    assert "Generating outputs" in log_text
    assert "Pipeline completed successfully" in log_text
    assert "Pseudo-Label Generation Summary" in log_text

