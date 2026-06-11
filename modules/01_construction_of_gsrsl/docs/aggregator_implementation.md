# Class Aggregator Implementation
# 类别聚合器实现

## Overview
## 概述

The class aggregator module (`gsrsl_pipeline/aggregator.py`) provides functionality to compute mean reference spectra for each lithology class by aggregating multiple resampled ground spectra. This is a critical step in building the Ground Standard Reference Spectral Library (GSRSL).

类别聚合器模块提供了通过聚合多个重采样地面光谱来计算每个岩性类别平均参考光谱的功能。这是构建地面标准参考光谱库(GSRSL)的关键步骤。

## Key Features
## 主要特性

### 1. Robust Input Validation
### 1. 健壮的输入验证

The `aggregate_by_class()` function performs comprehensive validation:
- Validates input is a list of tuples
- Checks each tuple has (class_id, spectrum_array) structure
- Validates class_id is an integer in range [0, 4]
- Validates spectrum arrays have shape (297,) and are numpy arrays
- Ensures all 5 lithology classes have at least one sample

aggregate_by_class()函数执行全面的验证：
- 验证输入是元组列表
- 检查每个元组具有(class_id, spectrum_array)结构
- 验证class_id是[0, 4]范围内的整数
- 验证光谱数组形状为(297,)且为numpy数组
- 确保所有5个岩性类别至少有一个样本

### 2. NaN Handling
### 2. NaN处理

The aggregator uses `np.nanmean()` to handle NaN values gracefully:
- **Partial NaN**: If some samples have NaN for a band, those values are ignored and the mean is computed from valid values only
- **All NaN**: If all samples have NaN for a band, the result is NaN for that band
- **Warning**: Logs a warning if any class has > 10% NaN bands in the final result

聚合器使用np.nanmean()优雅地处理NaN值：
- **部分NaN**：如果某些样本在某个波段有NaN，这些值被忽略，仅从有效值计算平均值
- **全部NaN**：如果所有样本在某个波段都是NaN，该波段的结果为NaN
- **警告**：如果任何类别在最终结果中有>10%的NaN波段，则记录警告

### 3. Class Independence
### 3. 类别独立性

Each lithology class is processed independently. Adding or modifying samples for one class does not affect the mean spectra computed for other classes.

每个岩性类别独立处理。添加或修改一个类别的样本不会影响为其他类别计算的平均光谱。

### 4. Comprehensive Logging
### 4. 全面的日志记录

The function logs detailed information about the aggregation process:
- Number of samples per class
- Number and percentage of NaN bands per class
- Warnings for classes with high NaN percentages
- Summary of total samples processed

函数记录有关聚合过程的详细信息：
- 每个类别的样本数
- 每个类别的NaN波段数量和百分比
- 对NaN百分比高的类别的警告
- 处理的总样本数摘要

## Usage Example
## 使用示例

```python
from gsrsl_pipeline.aggregator import aggregate_by_class
import numpy as np

# Prepare resampled spectra (from SpectralResampler)
resampled_spectra = [
    (0, np.full(297, 0.3, dtype=np.float32)),  # Class 0, sample 1
    (0, np.full(297, 0.4, dtype=np.float32)),  # Class 0, sample 2
    (1, np.full(297, 0.5, dtype=np.float32)),  # Class 1, sample 1
    (2, np.full(297, 0.6, dtype=np.float32)),  # Class 2, sample 1
    (3, np.full(297, 0.7, dtype=np.float32)),  # Class 3, sample 1
    (4, np.full(297, 0.8, dtype=np.float32)),  # Class 4, sample 1
]

# Aggregate by class
class_means = aggregate_by_class(resampled_spectra)

# Access mean spectrum for each class
for class_id in range(5):
    mean_spectrum = class_means[class_id]
    print(f"Class {class_id}: shape={mean_spectrum.shape}, dtype={mean_spectrum.dtype}")
```

## Integration with Pipeline
## 与管道集成

The aggregator fits into the GSRSL pipeline as follows:

聚合器在GSRSL管道中的位置如下：

```
Ground Spectra → Savitzky-Golay Filter → Spectral Resampler → Class Aggregator → GSRSL Output
地面光谱 → Savitzky-Golay滤波器 → 光谱重采样器 → 类别聚合器 → GSRSL输出
```

1. **Input**: List of (class_id, resampled_spectrum) tuples from SpectralResampler
2. **Processing**: Groups by class_id and computes mean using np.nanmean
3. **Output**: Dictionary mapping class_id (0-4) to mean spectrum arrays (297,)

1. **输入**：来自SpectralResampler的(class_id, resampled_spectrum)元组列表
2. **处理**：按class_id分组并使用np.nanmean计算平均值
3. **输出**：将class_id(0-4)映射到平均光谱数组(297,)的字典

## Requirements Satisfied
## 满足的需求

The aggregator module satisfies the following requirements from the specification:

聚合器模块满足规范中的以下需求：

- **Requirement 6.1**: Computes arithmetic mean across all samples for each class
- **Requirement 6.2**: Handles each of the five lithology classes independently
- **Requirement 6.3**: Ignores NaN values when computing the mean for each band
- **Requirement 6.4**: Sets mean to NaN if all samples for a band contain NaN
- **Requirement 6.5**: Returns dictionary mapping class_id (0-4) to numpy arrays of shape (297,) with dtype float32

- **需求6.1**：计算每个类别所有样本的算术平均值
- **需求6.2**：独立处理五个岩性类别中的每一个
- **需求6.3**：在计算每个波段的平均值时忽略NaN值
- **需求6.4**：如果波段的所有样本都包含NaN，则将平均值设置为NaN
- **需求6.5**：返回将class_id(0-4)映射到形状为(297,)、dtype为float32的numpy数组的字典

## Testing
## 测试

The module includes comprehensive unit tests (`tests/test_aggregator.py`) covering:

模块包含全面的单元测试，涵盖：

### Validation Tests (9 tests)
### 验证测试（9个测试）
- Invalid input types
- Empty lists
- Invalid tuple structures
- Invalid class_id types and ranges
- Invalid spectrum types and shapes
- Missing class samples

### Basic Functionality Tests (3 tests)
### 基本功能测试（3个测试）
- Single sample per class
- Multiple samples per class
- Varying number of samples per class

### NaN Handling Tests (4 tests)
### NaN处理测试（4个测试）
- NaN values ignored in mean computation
- All-NaN bands result in NaN mean
- Partial NaN bands across samples
- High NaN percentage warning

### Independence Tests (1 test)
### 独立性测试（1个测试）
- Class independence verification

### Realistic Tests (4 tests)
### 真实场景测试（4个测试）
- Realistic varying spectra
- Mixed NaN and valid values
- Output dtype preservation
- Numpy integer class_id support

**Total: 21 tests, all passing**
**总计：21个测试，全部通过**

## Performance Considerations
## 性能考虑

- Uses `np.stack()` for efficient array stacking
- Uses `np.nanmean()` for vectorized mean computation
- Memory efficient: processes one class at a time
- Typical processing time: < 1ms for 10 samples across 5 classes

- 使用np.stack()进行高效的数组堆叠
- 使用np.nanmean()进行矢量化平均值计算
- 内存高效：一次处理一个类别
- 典型处理时间：5个类别的10个样本<1ms

## Error Messages
## 错误消息

The function provides clear, descriptive error messages:

函数提供清晰、描述性的错误消息：

```python
# Example error messages
"resampled_spectra must be a list, got dict"
"resampled_spectra cannot be empty"
"class_id must be in range [0, 4], got 5 at index 3"
"Spectrum must have shape (297,), got (100,) at index 2"
"No samples found for class 2 (Mixed-type Rich Pegmatite). Each class must have at least one sample."
```

## Lithology Classes
## 岩性类别

The five lithology classes supported:

支持的五个岩性类别：

| Class ID | English Name | Chinese Name |
|----------|--------------|--------------|
| 0 | Spodumene-rich Pegmatite | 锂辉石型富矿伟晶岩 |
| 1 | Lepidolite-rich Pegmatite | 锂云母型富矿伟晶岩 |
| 2 | Mixed-type Rich Pegmatite | 混合型富矿伟晶岩 |
| 3 | Barren Pegmatite | 贫矿伟晶岩 |
| 4 | Wall Rock | 围岩 |

## Demo Script
## 演示脚本

A complete demonstration is available in `examples/demo_aggregator.py` showing:
- Creating sample ground spectra
- Applying Savitzky-Golay filtering
- Resampling to satellite bands
- Aggregating by class
- Displaying results with statistics

完整的演示可在examples/demo_aggregator.py中找到，展示：
- 创建样本地面光谱
- 应用Savitzky-Golay滤波
- 重采样到卫星波段
- 按类别聚合
- 显示带统计信息的结果

Run the demo:
```bash
python examples/demo_aggregator.py
```

## Next Steps
## 下一步

After aggregation, the class mean spectra are ready for:
1. Saving to `gsrsl.npy` file using `numpy.save()`
2. Visualization using matplotlib
3. Integration with deep learning models (AG-S²CNN)

聚合后，类别平均光谱准备好用于：
1. 使用numpy.save()保存到gsrsl.npy文件
2. 使用matplotlib进行可视化
3. 与深度学习模型(AG-S²CNN)集成
