"""
Integration tests for ReferenceSpectraLoader with InputValidator.
ReferenceSpectraLoader与InputValidator的集成测试。
"""

import pytest
import numpy as np
import tempfile
import os
from hyperspectral_pseudo_label_generator.input import ReferenceSpectraLoader, InputValidator


class TestReferenceSpectraLoaderIntegration:
    """Integration tests for ReferenceSpectraLoader with InputValidator."""
    
    def test_loaded_spectra_pass_validation(self):
        """
        Test that spectra loaded by ReferenceSpectraLoader pass InputValidator validation.
        测试ReferenceSpectraLoader加载的光谱通过InputValidator验证。
        
        Validates: Requirements 2.1, 2.2, 2.4, 2.5
        """
        # Create valid reference spectra
        # 创建有效的参考光谱
        spectra = np.random.rand(5, 297).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            # Load spectra
            # 加载光谱
            loaded_spectra = ReferenceSpectraLoader.load(temp_path)
            
            # Validate - should not raise any exception
            # 验证 - 不应引发任何异常
            InputValidator.validate_reference_spectra(loaded_spectra)
            
        finally:
            os.unlink(temp_path)
    
    def test_invalid_spectra_fail_validation(self):
        """
        Test that invalid spectra fail InputValidator validation.
        测试无效光谱未通过InputValidator验证。
        
        Validates: Requirements 2.3, 2.5
        """
        # Create spectra with NaN values
        # 创建包含NaN值的光谱
        spectra = np.random.rand(5, 297).astype(np.float32)
        spectra[2, 100] = np.nan
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            # Loading should fail before validation
            # 加载应在验证之前失败
            with pytest.raises(ValueError) as exc_info:
                ReferenceSpectraLoader.load(temp_path)
            
            assert "NaN" in str(exc_info.value)
            
        finally:
            os.unlink(temp_path)
    
    def test_validator_catches_wrong_band_count(self):
        """
        Test that InputValidator catches wrong band count.
        测试InputValidator捕获错误的波段计数。
        
        Validates: Requirements 2.2, 2.3
        """
        # Create manually constructed dict with wrong band count
        # 创建手动构造的具有错误波段计数的字典
        invalid_spectra = {
            i: np.random.rand(250).astype(np.float32) for i in range(5)
        }
        
        with pytest.raises(ValueError) as exc_info:
            InputValidator.validate_reference_spectra(invalid_spectra)
        
        assert "expected 297 bands" in str(exc_info.value).lower()
    
    def test_validator_catches_wrong_class_count(self):
        """
        Test that InputValidator catches wrong class count.
        测试InputValidator捕获错误的类别计数。
        
        Validates: Requirements 2.1, 2.3
        """
        # Create manually constructed dict with wrong class count
        # 创建手动构造的具有错误类别计数的字典
        invalid_spectra = {
            i: np.random.rand(297).astype(np.float32) for i in range(3)
        }
        
        with pytest.raises(ValueError) as exc_info:
            InputValidator.validate_reference_spectra(invalid_spectra)
        
        assert "Expected 5 classes, got 3" in str(exc_info.value)
    
    def test_validator_catches_nan_in_spectra(self):
        """
        Test that InputValidator catches NaN values in spectra.
        测试InputValidator捕获光谱中的NaN值。
        
        Validates: Requirements 2.5, 2.3
        """
        # Create dict with NaN values
        # 创建包含NaN值的字典
        invalid_spectra = {
            i: np.random.rand(297).astype(np.float32) for i in range(5)
        }
        invalid_spectra[2][100] = np.nan
        
        with pytest.raises(ValueError) as exc_info:
            InputValidator.validate_reference_spectra(invalid_spectra)
        
        assert "Class 2" in str(exc_info.value)
        assert "NaN" in str(exc_info.value)
    
    def test_validator_catches_inf_in_spectra(self):
        """
        Test that InputValidator catches Inf values in spectra.
        测试InputValidator捕获光谱中的Inf值。
        
        Validates: Requirements 2.5, 2.3
        """
        # Create dict with Inf values
        # 创建包含Inf值的字典
        invalid_spectra = {
            i: np.random.rand(297).astype(np.float32) for i in range(5)
        }
        invalid_spectra[4][50] = np.inf
        
        with pytest.raises(ValueError) as exc_info:
            InputValidator.validate_reference_spectra(invalid_spectra)
        
        assert "Class 4" in str(exc_info.value)
        assert "Inf" in str(exc_info.value)
    
    def test_end_to_end_load_and_validate(self):
        """
        Test complete end-to-end workflow: load and validate reference spectra.
        测试完整的端到端工作流：加载和验证参考光谱。
        
        Validates: Requirements 2.1, 2.2, 2.4, 2.5
        """
        # Create valid reference spectra with realistic values
        # 创建具有真实值的有效参考光谱
        # Simulate reflectance values in range [0, 1]
        # 模拟范围[0, 1]的反射率值
        spectra = np.random.uniform(0.0, 1.0, (5, 297)).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            # Step 1: Load spectra
            # 步骤1: 加载光谱
            loaded_spectra = ReferenceSpectraLoader.load(temp_path)
            
            # Step 2: Validate spectra
            # 步骤2: 验证光谱
            InputValidator.validate_reference_spectra(loaded_spectra)
            
            # Step 3: Verify structure
            # 步骤3: 验证结构
            assert len(loaded_spectra) == 5
            for i in range(5):
                assert i in loaded_spectra
                assert loaded_spectra[i].shape == (297,)
                assert np.all(np.isfinite(loaded_spectra[i]))
                assert np.all(loaded_spectra[i] >= 0)
                assert np.all(loaded_spectra[i] <= 1)
            
        finally:
            os.unlink(temp_path)
