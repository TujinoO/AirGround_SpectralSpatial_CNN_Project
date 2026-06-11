"""
Integration tests for the complete input module.
输入模块的集成测试。

This test suite verifies that all input module components (HyperspectralImageLoader,
ReferenceSpectraLoader, MetadataParser, InputValidator) work correctly together.
此测试套件验证所有输入模块组件是否正确协同工作。
"""

import numpy as np
import pytest
import tempfile
import os
import json

from hyperspectral_pseudo_label_generator.input.image_loader import HyperspectralImageLoader
from hyperspectral_pseudo_label_generator.input.reference_loader import ReferenceSpectraLoader
from hyperspectral_pseudo_label_generator.input.metadata_parser import MetadataParser
from hyperspectral_pseudo_label_generator.input.validator import InputValidator
from hyperspectral_pseudo_label_generator import ProcessingConfig


class TestInputModuleIntegration:
    """Integration tests for the complete input module workflow."""
    
    def test_complete_input_workflow_npy_format(self):
        """
        Test complete workflow: load image, reference spectra, metadata, and validate all.
        测试完整工作流：加载影像、参考光谱、元数据并验证所有内容。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic GF-5 data
            H, W = 50, 50
            image = np.random.rand(H, W, 297).astype(np.float32)
            image_path = os.path.join(tmpdir, "test_image.npy")
            np.save(image_path, image)
            
            # Create reference spectra
            reference_spectra = np.random.rand(5, 297).astype(np.float32)
            reference_path = os.path.join(tmpdir, "reference_spectra.npy")
            np.save(reference_path, reference_spectra)
            
            # Create wavelength metadata
            wavelengths = np.linspace(400, 2500, 297)
            metadata_path = os.path.join(tmpdir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({"wavelengths": wavelengths.tolist()}, f)
            
            # Load all data
            loaded_image = HyperspectralImageLoader.load(image_path)
            loaded_spectra = ReferenceSpectraLoader.load(reference_path)
            loaded_wavelengths = MetadataParser.parse_wavelengths(metadata_path)
            
            # Validate all data
            InputValidator.validate_image(loaded_image)
            InputValidator.validate_reference_spectra(loaded_spectra)
            InputValidator.validate_wavelengths(loaded_wavelengths)
            
            # Verify shapes and values
            assert loaded_image.shape == (H, W, 297)
            assert len(loaded_spectra) == 5
            assert loaded_wavelengths.shape == (297,)
            assert np.allclose(loaded_image, image)
            assert np.allclose(loaded_wavelengths, wavelengths)
    
    def test_complete_input_workflow_csv_format(self):
        """
        Test complete workflow with CSV format for reference spectra and metadata.
        使用CSV格式测试参考光谱和元数据的完整工作流。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic GF-5 data
            H, W = 30, 30
            image = np.random.rand(H, W, 297).astype(np.float32)
            image_path = os.path.join(tmpdir, "test_image.npy")
            np.save(image_path, image)
            
            # Create reference spectra as CSV
            reference_spectra = np.random.rand(5, 297).astype(np.float32)
            reference_path = os.path.join(tmpdir, "reference_spectra.csv")
            np.savetxt(reference_path, reference_spectra, delimiter=',')
            
            # Create wavelength metadata as CSV
            wavelengths = np.linspace(400, 2500, 297)
            metadata_path = os.path.join(tmpdir, "metadata.csv")
            np.savetxt(metadata_path, wavelengths, delimiter=',')
            
            # Load all data
            loaded_image = HyperspectralImageLoader.load(image_path)
            loaded_spectra = ReferenceSpectraLoader.load(reference_path)
            loaded_wavelengths = MetadataParser.parse_wavelengths(metadata_path)
            
            # Validate all data
            InputValidator.validate_image(loaded_image)
            InputValidator.validate_reference_spectra(loaded_spectra)
            InputValidator.validate_wavelengths(loaded_wavelengths)
            
            # Verify all validations passed
            assert loaded_image.shape == (H, W, 297)
            assert len(loaded_spectra) == 5
            assert loaded_wavelengths.shape == (297,)
    
    def test_aloh_band_identification_with_real_gf5_wavelengths(self):
        """
        Test Al-OH band identification with realistic GF-5 wavelength distribution.
        使用真实的GF-5波长分布测试Al-OH波段识别。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create realistic GF-5 wavelengths (400-2500nm range)
            # GF-5 has higher density in certain regions
            wavelengths = np.linspace(400, 2500, 297)
            metadata_path = os.path.join(tmpdir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({"wavelengths": wavelengths.tolist()}, f)
            
            # Load and validate wavelengths
            loaded_wavelengths = MetadataParser.parse_wavelengths(metadata_path)
            InputValidator.validate_wavelengths(loaded_wavelengths)
            
            # Identify Al-OH absorption band
            config = ProcessingConfig()
            aloh_mask, aloh_indices = MetadataParser.identify_aloh_band(
                loaded_wavelengths,
                config.aloh_min_wavelength,
                config.aloh_max_wavelength
            )
            
            # Verify Al-OH band is correctly identified
            assert len(aloh_indices) > 0, "Al-OH band should be identified"
            aloh_wavelengths = loaded_wavelengths[aloh_indices]
            assert np.all(aloh_wavelengths >= config.aloh_min_wavelength)
            assert np.all(aloh_wavelengths <= config.aloh_max_wavelength)
    
    def test_invalid_image_caught_by_validator(self):
        """
        Test that invalid images are caught by the validator.
        测试验证器能够捕获无效影像。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid image (wrong band count)
            invalid_image = np.random.rand(50, 50, 250).astype(np.float32)
            image_path = os.path.join(tmpdir, "invalid_image.npy")
            np.save(image_path, invalid_image)
            
            # Load image
            loaded_image = HyperspectralImageLoader.load(image_path)
            
            # Validation should fail
            with pytest.raises(ValueError, match="Expected 297 bands"):
                InputValidator.validate_image(loaded_image)
    
    def test_invalid_reference_spectra_caught_by_validator(self):
        """
        Test that invalid reference spectra are caught by the validator.
        测试验证器能够捕获无效参考光谱。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid reference spectra (wrong class count)
            invalid_spectra = np.random.rand(3, 297).astype(np.float32)
            reference_path = os.path.join(tmpdir, "invalid_spectra.npy")
            np.save(reference_path, invalid_spectra)
            
            # Loading should fail (ReferenceSpectraLoader validates on load)
            with pytest.raises(ValueError, match="Expected 5 classes"):
                ReferenceSpectraLoader.load(reference_path)
    
    def test_invalid_wavelengths_caught_by_validator(self):
        """
        Test that invalid wavelengths are caught by the validator.
        测试验证器能够捕获无效波长。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid wavelengths (contains NaN)
            invalid_wavelengths = np.linspace(400, 2500, 297).astype(float)
            invalid_wavelengths[0] = np.nan
            metadata_path = os.path.join(tmpdir, "invalid_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({"wavelengths": invalid_wavelengths.tolist()}, f)
            
            # Load wavelengths
            loaded_wavelengths = MetadataParser.parse_wavelengths(metadata_path)
            
            # Validation should fail
            with pytest.raises(ValueError, match="NaN or Inf"):
                InputValidator.validate_wavelengths(loaded_wavelengths)
    
    def test_end_to_end_with_config_validation(self):
        """
        Test end-to-end workflow including configuration validation.
        测试包括配置验证的端到端工作流。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid configuration
            config = ProcessingConfig(
                n_components=5,
                ore_percentile=2.5,
                non_ore_percentile=5.0,
                ambiguity_threshold=0.1,
                output_dir=tmpdir
            )
            
            # Validate configuration
            InputValidator.validate_config(config)
            
            # Create synthetic data
            H, W = 40, 40
            image = np.random.rand(H, W, 297).astype(np.float32)
            image_path = os.path.join(tmpdir, "test_image.npy")
            np.save(image_path, image)
            
            reference_spectra = np.random.rand(5, 297).astype(np.float32)
            reference_path = os.path.join(tmpdir, "reference_spectra.npy")
            np.save(reference_path, reference_spectra)
            
            wavelengths = np.linspace(400, 2500, 297)
            metadata_path = os.path.join(tmpdir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({"wavelengths": wavelengths.tolist()}, f)
            
            # Load and validate all data
            loaded_image = HyperspectralImageLoader.load(image_path)
            loaded_spectra = ReferenceSpectraLoader.load(reference_path)
            loaded_wavelengths = MetadataParser.parse_wavelengths(metadata_path)
            
            InputValidator.validate_image(loaded_image)
            InputValidator.validate_reference_spectra(loaded_spectra)
            InputValidator.validate_wavelengths(loaded_wavelengths)
            
            # Verify Al-OH band can be identified
            aloh_mask, aloh_indices = MetadataParser.identify_aloh_band(
                loaded_wavelengths,
                config.aloh_min_wavelength,
                config.aloh_max_wavelength
            )
            
            assert len(aloh_indices) > 0
            assert loaded_image.shape == (H, W, 297)
            assert len(loaded_spectra) == 5
            assert loaded_wavelengths.shape == (297,)
    
    def test_data_consistency_across_formats(self):
        """
        Test that data loaded from different formats is consistent.
        测试从不同格式加载的数据是一致的。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create reference data
            reference_spectra = np.random.rand(5, 297).astype(np.float32)
            
            # Save as NPY
            npy_path = os.path.join(tmpdir, "spectra.npy")
            np.save(npy_path, reference_spectra)
            
            # Save as CSV
            csv_path = os.path.join(tmpdir, "spectra.csv")
            np.savetxt(csv_path, reference_spectra, delimiter=',')
            
            # Load from both formats
            loaded_npy = ReferenceSpectraLoader.load(npy_path)
            loaded_csv = ReferenceSpectraLoader.load(csv_path)
            
            # Verify consistency
            for class_id in range(5):
                assert np.allclose(loaded_npy[class_id], loaded_csv[class_id], rtol=1e-5)
    
    def test_large_image_handling(self):
        """
        Test that the input module can handle large images.
        测试输入模块能够处理大型影像。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a larger image (simulating real GF-5 data)
            H, W = 500, 500
            image = np.random.rand(H, W, 297).astype(np.float32)
            image_path = os.path.join(tmpdir, "large_image.npy")
            np.save(image_path, image)
            
            # Load and validate
            loaded_image = HyperspectralImageLoader.load(image_path)
            InputValidator.validate_image(loaded_image)
            
            assert loaded_image.shape == (H, W, 297)
            assert loaded_image.dtype == np.float32
    
    def test_edge_case_single_pixel_image(self):
        """
        Test handling of edge case: single pixel image.
        测试边缘情况：单像素影像。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create single pixel image
            image = np.random.rand(1, 1, 297).astype(np.float32)
            image_path = os.path.join(tmpdir, "single_pixel.npy")
            np.save(image_path, image)
            
            # Load and validate
            loaded_image = HyperspectralImageLoader.load(image_path)
            InputValidator.validate_image(loaded_image)
            
            assert loaded_image.shape == (1, 1, 297)
    
    def test_class_organization_in_reference_spectra(self):
        """
        Test that reference spectra are correctly organized by class.
        测试参考光谱按类别正确组织。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create reference spectra with distinct values per class
            reference_spectra = np.zeros((5, 297), dtype=np.float32)
            for class_id in range(5):
                reference_spectra[class_id, :] = class_id + 1.0
            
            reference_path = os.path.join(tmpdir, "reference_spectra.npy")
            np.save(reference_path, reference_spectra)
            
            # Load spectra
            loaded_spectra = ReferenceSpectraLoader.load(reference_path)
            
            # Validate
            InputValidator.validate_reference_spectra(loaded_spectra)
            
            # Verify organization
            assert len(loaded_spectra) == 5
            for class_id in range(5):
                assert class_id in loaded_spectra
                assert np.all(loaded_spectra[class_id] == class_id + 1.0)
