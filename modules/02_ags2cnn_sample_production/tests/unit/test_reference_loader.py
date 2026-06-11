"""
Unit tests for ReferenceSpectraLoader class.
ReferenceSpectraLoader类的单元测试。
"""

import pytest
import numpy as np
import tempfile
import os
from hyperspectral_pseudo_label_generator.input import ReferenceSpectraLoader


class TestReferenceSpectraLoader:
    """Test suite for ReferenceSpectraLoader class."""
    
    def test_load_valid_npy_file(self):
        """
        Test loading valid reference spectra from .npy file.
        测试从.npy文件加载有效的参考光谱。
        
        Validates: Requirements 2.1, 2.2, 2.4
        """
        # Create valid reference spectra (5 classes, 297 bands)
        # 创建有效的参考光谱 (5个类别, 297个波段)
        spectra = np.random.rand(5, 297).astype(np.float32)
        
        # Save to temporary .npy file
        # 保存到临时.npy文件
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            # Load spectra
            # 加载光谱
            result = ReferenceSpectraLoader.load(temp_path)
            
            # Verify structure
            # 验证结构
            assert isinstance(result, dict), "Result should be a dictionary"
            assert len(result) == 5, "Should have 5 classes"
            
            # Verify all classes are present
            # 验证所有类别都存在
            for i in range(5):
                assert i in result, f"Class {i} should be present"
                assert result[i].shape == (297,), f"Class {i} should have 297 bands"
                assert np.allclose(result[i], spectra[i]), f"Class {i} data should match"
        
        finally:
            # Clean up
            # 清理
            os.unlink(temp_path)
    
    def test_load_valid_csv_file(self):
        """
        Test loading valid reference spectra from .csv file.
        测试从.csv文件加载有效的参考光谱。
        
        Validates: Requirements 2.1, 2.2, 2.4
        """
        # Create valid reference spectra (5 classes, 297 bands)
        # 创建有效的参考光谱 (5个类别, 297个波段)
        spectra = np.random.rand(5, 297).astype(np.float32)
        
        # Save to temporary .csv file
        # 保存到临时.csv文件
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            temp_path = f.name
        
        np.savetxt(temp_path, spectra, delimiter=',')
        
        try:
            # Load spectra
            # 加载光谱
            result = ReferenceSpectraLoader.load(temp_path)
            
            # Verify structure
            # 验证结构
            assert isinstance(result, dict), "Result should be a dictionary"
            assert len(result) == 5, "Should have 5 classes"
            
            # Verify all classes are present
            # 验证所有类别都存在
            for i in range(5):
                assert i in result, f"Class {i} should be present"
                assert result[i].shape == (297,), f"Class {i} should have 297 bands"
                assert np.allclose(result[i], spectra[i], rtol=1e-5), f"Class {i} data should match"
        
        finally:
            # Clean up
            # 清理
            os.unlink(temp_path)
    
    def test_load_incorrect_number_of_classes(self):
        """
        Test that loading spectra with incorrect number of classes raises ValueError.
        测试加载类别数量不正确的光谱会引发ValueError。
        
        Validates: Requirements 2.1, 2.3
        """
        # Create spectra with wrong number of classes (3 instead of 5)
        # 创建类别数量错误的光谱 (3个而不是5个)
        spectra = np.random.rand(3, 297).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ReferenceSpectraLoader.load(temp_path)
            
            assert "Expected 5 classes, got 3" in str(exc_info.value)
        
        finally:
            os.unlink(temp_path)
    
    def test_load_incorrect_number_of_bands(self):
        """
        Test that loading spectra with incorrect number of bands raises ValueError.
        测试加载波段数量不正确的光谱会引发ValueError。
        
        Validates: Requirements 2.2, 2.3
        """
        # Create spectra with wrong number of bands (250 instead of 297)
        # 创建波段数量错误的光谱 (250个而不是297个)
        spectra = np.random.rand(5, 250).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ReferenceSpectraLoader.load(temp_path)
            
            assert "Expected 297 bands, got 250" in str(exc_info.value)
        
        finally:
            os.unlink(temp_path)
    
    def test_load_spectra_with_nan_values(self):
        """
        Test that loading spectra with NaN values raises ValueError.
        测试加载包含NaN值的光谱会引发ValueError。
        
        Validates: Requirements 2.5, 2.3
        """
        # Create spectra with NaN values
        # 创建包含NaN值的光谱
        spectra = np.random.rand(5, 297).astype(np.float32)
        spectra[2, 100] = np.nan  # Add NaN value
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ReferenceSpectraLoader.load(temp_path)
            
            assert "NaN" in str(exc_info.value)
        
        finally:
            os.unlink(temp_path)
    
    def test_load_spectra_with_inf_values(self):
        """
        Test that loading spectra with Inf values raises ValueError.
        测试加载包含Inf值的光谱会引发ValueError。
        
        Validates: Requirements 2.5, 2.3
        """
        # Create spectra with Inf values
        # 创建包含Inf值的光谱
        spectra = np.random.rand(5, 297).astype(np.float32)
        spectra[1, 50] = np.inf  # Add Inf value
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ReferenceSpectraLoader.load(temp_path)
            
            assert "Inf" in str(exc_info.value)
        
        finally:
            os.unlink(temp_path)
    
    def test_load_unsupported_file_format(self):
        """
        Test that loading from unsupported file format raises ValueError.
        测试从不支持的文件格式加载会引发ValueError。
        
        Validates: Requirements 2.3
        """
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
            f.write(b"dummy data")
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ReferenceSpectraLoader.load(temp_path)
            
            assert "Unsupported file format" in str(exc_info.value)
            assert ".npy and .csv" in str(exc_info.value)
        
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """
        Test that loading from non-existent file raises FileNotFoundError.
        测试从不存在的文件加载会引发FileNotFoundError。
        
        Validates: Requirements 2.3
        """
        nonexistent_path = "/tmp/nonexistent_reference_spectra_12345.npy"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            ReferenceSpectraLoader.load(nonexistent_path)
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_load_malformed_csv_file(self):
        """
        Test that loading from malformed CSV file raises IOError.
        测试从格式错误的CSV文件加载会引发IOError。
        
        Validates: Requirements 2.3
        """
        # Create malformed CSV file
        # 创建格式错误的CSV文件
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            temp_path = f.name
            f.write("not,a,valid,csv,with,proper,structure\n")
            f.write("this,is,malformed,data\n")
        
        try:
            with pytest.raises(IOError) as exc_info:
                ReferenceSpectraLoader.load(temp_path)
            
            assert "Failed to load .csv file" in str(exc_info.value)
        
        finally:
            os.unlink(temp_path)
    
    def test_load_1d_array(self):
        """
        Test that loading 1D array raises ValueError.
        测试加载1D数组会引发ValueError。
        
        Validates: Requirements 2.3
        """
        # Create 1D array instead of 2D
        # 创建1D数组而不是2D
        spectra = np.random.rand(297).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ReferenceSpectraLoader.load(temp_path)
            
            assert "must be 2D array" in str(exc_info.value)
        
        finally:
            os.unlink(temp_path)
    
    def test_load_3d_array(self):
        """
        Test that loading 3D array raises ValueError.
        测试加载3D数组会引发ValueError。
        
        Validates: Requirements 2.3
        """
        # Create 3D array instead of 2D
        # 创建3D数组而不是2D
        spectra = np.random.rand(5, 297, 10).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ReferenceSpectraLoader.load(temp_path)
            
            assert "must be 2D array" in str(exc_info.value)
        
        finally:
            os.unlink(temp_path)
    
    def test_class_organization(self):
        """
        Test that classes are properly organized by index 0-4.
        测试类别按索引0-4正确组织。
        
        Validates: Requirements 2.4
        
        Class organization:
        - Classes 0-2: Ore classes (矿石类别)
        - Class 3: Barren Pegmatite (贫矿伟晶岩)
        - Class 4: Wall Rock (围岩)
        """
        # Create reference spectra with distinct values for each class
        # 为每个类别创建具有不同值的参考光谱
        spectra = np.zeros((5, 297), dtype=np.float32)
        for i in range(5):
            spectra[i, :] = float(i)  # Each class has unique constant value
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            result = ReferenceSpectraLoader.load(temp_path)
            
            # Verify organization
            # 验证组织
            assert set(result.keys()) == {0, 1, 2, 3, 4}, "Should have classes 0-4"
            
            # Verify each class has correct data
            # 验证每个类别有正确的数据
            for i in range(5):
                assert np.all(result[i] == float(i)), f"Class {i} should have value {i}"
        
        finally:
            os.unlink(temp_path)
    
    def test_load_preserves_dtype(self):
        """
        Test that loading preserves the data type of the spectra.
        测试加载保留光谱的数据类型。
        
        Validates: Requirements 2.1
        """
        # Create spectra with float32 dtype
        # 创建float32数据类型的光谱
        spectra = np.random.rand(5, 297).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            result = ReferenceSpectraLoader.load(temp_path)
            
            # Verify dtype is preserved
            # 验证数据类型被保留
            for i in range(5):
                assert result[i].dtype == np.float32, f"Class {i} should have float32 dtype"
        
        finally:
            os.unlink(temp_path)
    
    def test_load_edge_case_all_zeros(self):
        """
        Test loading spectra with all zero values (edge case).
        测试加载所有值为零的光谱（边缘情况）。
        
        Validates: Requirements 2.1, 2.2
        """
        # Create spectra with all zeros
        # 创建所有值为零的光谱
        spectra = np.zeros((5, 297), dtype=np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            result = ReferenceSpectraLoader.load(temp_path)
            
            # Should load successfully
            # 应该成功加载
            assert len(result) == 5
            for i in range(5):
                assert np.all(result[i] == 0), f"Class {i} should be all zeros"
        
        finally:
            os.unlink(temp_path)
    
    def test_load_edge_case_very_large_values(self):
        """
        Test loading spectra with very large values (edge case).
        测试加载具有非常大值的光谱（边缘情况）。
        
        Validates: Requirements 2.1, 2.2
        """
        # Create spectra with very large values
        # 创建具有非常大值的光谱
        spectra = np.full((5, 297), 1e6, dtype=np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
            np.save(temp_path, spectra)
        
        try:
            result = ReferenceSpectraLoader.load(temp_path)
            
            # Should load successfully
            # 应该成功加载
            assert len(result) == 5
            for i in range(5):
                assert np.all(result[i] == 1e6), f"Class {i} should have value 1e6"
        
        finally:
            os.unlink(temp_path)
