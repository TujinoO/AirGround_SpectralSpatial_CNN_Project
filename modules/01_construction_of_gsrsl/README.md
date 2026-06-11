# GSRSL Pipeline
# GSRSL管道

Ground Standard Reference Spectral Library (GSRSL) data processing pipeline for hyperspectral remote sensing and deep learning applications.

地面标准参考光谱库(GSRSL)数据处理管道，用于高光谱遥感和深度学习应用。

## Overview
## 概述

This pipeline processes high-resolution ground spectral data from TSG8 software and resamples it to match GF-5 satellite specifications for lithium pegmatite prediction using air-ground synergistic deep learning.

该管道处理来自TSG8软件的高分辨率地面光谱数据，并将其重采样以匹配GF-5卫星规格，用于使用空地协同深度学习进行锂伟晶岩预测。

## Project Structure
## 项目结构

```
gsrsl-pipeline/
├── gsrsl_pipeline/          # Main package / 主包
│   ├── __init__.py
│   └── logging_config.py    # Logging configuration / 日志配置
├── tests/                   # Test suite / 测试套件
│   └── __init__.py
├── data/                    # Data directory / 数据目录
├── examples/                # Example scripts / 示例脚本
├── requirements.txt         # Python dependencies / Python依赖项
├── setup.py                 # Package installation / 包安装
└── README.md               # This file / 本文件
```

## Installation
## 安装

### Prerequisites
### 前提条件

- Python 3.8 or higher
- pip package manager

### Install Dependencies
### 安装依赖项

```bash
pip install -r requirements.txt
```

### Install Package
### 安装包

For development:
```bash
pip install -e .
```

For production:
```bash
pip install .
```

## Dependencies
## 依赖项

- **numpy**: Numerical computing / 数值计算
- **pandas**: Data processing / 数据处理
- **scipy**: Scientific computing (Savitzky-Golay filter) / 科学计算（Savitzky-Golay滤波器）
- **matplotlib**: Visualization / 可视化
- **pytest**: Unit testing / 单元测试
- **hypothesis**: Property-based testing / 属性测试

## Usage
## 使用

### Quick Start
### 快速开始

The GSRSL pipeline is executed using the `build_gsrsl.py` script:

GSRSL管道使用`build_gsrsl.py`脚本执行：

```bash
python build_gsrsl.py \
    --metadata D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/1_gsrsl/GF5A_AHSI_20240126__metadata.txt \
    --spectra D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/1_gsrsl/spectra.csv \
    --labels D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/1_gsrsl/labels.csv \
    --output D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/1_gsrsl/output_Presentation/
```

### Command-Line Arguments
### 命令行参数

**Required Arguments / 必需参数:**

- `--metadata`: Path to GF-5 metadata text file containing wavelengths and FWHM values  
  - GF-5元数据文本文件的路径，包含波长和FWHM值
- `--spectra`: Path to **matrix-format** CSV file containing ground spectra from TSG8  
  - 地面光谱矩阵 CSV 文件的路径（来自 TSG8）：第 1 列为波长，第 2 列及之后每列为一个样品的反射率序列
- `--labels`: Path to CSV file mapping spectrum column names to class IDs (columns: filename, class_id)  
  - 标签 CSV 文件的路径，将光谱列名映射到岩性类别 ID（列：filename, class_id）
- `--output`: Path to output directory for GSRSL file and visualization  
  - GSRSL文件和可视化的输出目录路径

**Optional Arguments / 可选参数:**

- `--log`: Path to log file (default: gsrsl_pipeline.log)
  - 日志文件的路径（默认：gsrsl_pipeline.log）
- `--verbose`: Enable verbose logging (DEBUG level)
  - 启用详细日志记录（DEBUG级别）
- `--pattern`: File pattern for spectrum files (default: *.csv)
  - 光谱文件的文件模式（默认：*.csv）

### Input File Formats
### 输入文件格式

#### 1. GF-5 Metadata File
#### 1. GF-5元数据文件

Text file with wavelengths and FWHM values for 297 bands:

包含297个波段的波长和FWHM值的文本文件：

```
Wavelengths 1 = 387.21
FWHM 1 = 4.38
Wavelengths 2 = 394.21
FWHM 2 = 4.39
...
Wavelengths 297 = 2464.21
FWHM 297 = 7.34
```

#### 2. Ground Spectra Matrix File
#### 2. 地面光谱矩阵文件

Single CSV file with one wavelength column and multiple spectrum columns:

单个 CSV 文件，包含 1 列波长和多列光谱数据：

```csv
Wavelength_(nm),000001:hll.001,000002:hll.002,000003:hll.003,...
350.0,0.234,0.241,0.238,...
351.0,0.236,0.243,0.240,...
352.0,0.238,0.245,0.242,...
...
2500.0,0.456,0.462,0.459,...
```

- Column 1 (`Wavelength_(nm)`): Wavelengths in nanometers (typically 350–2500 nm, 1 nm step)  
  - 第 1 列 `Wavelength_(nm)`：波长，单位为 nm（通常 350–2500 nm，步长 1 nm）
- Columns 2..N: Reflectance values for each sample spectrum (range [0.0, 1.0])  
  - 第 2 列到第 N 列：各样品的反射率光谱（取值范围 [0.0, 1.0]）

The column names (e.g. `000001:hll.001`) must match the `filename` entries in the label table.  
这些光谱列名（如 `000001:hll.001`）必须与标签表中 `filename` 列的取值一一对应。

#### 3. Label Table
#### 3. 标签表

CSV file mapping spectrum column names to lithology class IDs:

将光谱列名映射到岩性类别ID的CSV文件：

```csv
filename,class_id
sample_001.csv,0
sample_002.csv,1
sample_003.csv,2
...
```

**Lithology Classes / 岩性类别:**
- 0: Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)
- 1: Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)
- 2: Mixed-type Rich Pegmatite (混合型富矿伟晶岩)
- 3: Barren Pegmatite (贫矿伟晶岩)
- 4: Wall Rock (围岩)

### Output Files
### 输出文件

The pipeline generates three output files:

管道生成三个输出文件：

1. **gsrsl.npy**: NumPy dictionary file containing mean reference spectra for all 5 classes
   - NumPy字典文件，包含所有5个类别的平均参考光谱
   - Format: `{0: array(297,), 1: array(297,), ..., 4: array(297,)}`
   - 格式：`{0: array(297,), 1: array(297,), ..., 4: array(297,)}`

2. **gsrsl_visualization.png**: Visualization plot showing all 5 reference spectra
   - 显示所有5个参考光谱的可视化图

3. **gsrsl_pipeline.log**: Detailed log file with processing information
   - 包含处理信息的详细日志文件

### Loading GSRSL Output
### 加载GSRSL输出

```python
import numpy as np

# Load GSRSL
gsrsl = np.load('output/gsrsl.npy', allow_pickle=True).item()

# Access class spectra
spodumene_spectrum = gsrsl[0]  # Class 0: Spodumene-rich
lepidolite_spectrum = gsrsl[1]  # Class 1: Lepidolite-rich
# ... etc

# Each spectrum is a numpy array of shape (297,) with dtype float32
print(spodumene_spectrum.shape)  # (297,)
print(spodumene_spectrum.dtype)  # float32
```

### Example: Complete Workflow
### 示例：完整工作流程

```bash
# 1. Prepare your data
# 1. 准备数据
# Create a matrix-format spectra CSV, e.g. data/ground_spectra_matrix.csv
# 创建一个矩阵格式的光谱 CSV，例如 data/ground_spectra_matrix.csv

# 2. Create label table
# 2. 创建标签表
# Create data/labels.csv with filename and class_id columns
# 创建包含 filename 和 class_id 列的 data/labels.csv（filename 对应光谱列名）

# 3. Run pipeline
# 3. 运行管道
python build_gsrsl.py --metadata D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/1_gsrsl/GF5A_AHSI_20240126__metadata.txt --spectra D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/1_gsrsl/spectra.csv --labels D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/1_gsrsl/labels.csv --output D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/1_gsrsl/output_Presentation/

# 4. Check outputs
# 4. 检查输出
ls -lh output/
# gsrsl.npy
# gsrsl_visualization.png
# gsrsl_pipeline.log
```

### Running the Demo
### 运行演示

A complete demo with synthetic data is provided:

提供了包含合成数据的完整演示：

```bash
python examples/demo_build_gsrsl.py
```

This demo:
- Creates synthetic ground spectra for all 5 lithology classes
- Runs the complete pipeline
- Verifies outputs
- Shows how to use the pipeline

该演示：
- 为所有5个岩性类别创建合成地面光谱
- 运行完整管道
- 验证输出
- 展示如何使用管道

### Pipeline Processing Steps
### 管道处理步骤

The pipeline performs the following steps:

管道执行以下步骤：

1. **Parse GF-5 Metadata**: Extract 297 band wavelengths and FWHM values
   - 解析GF-5元数据：提取297个波段波长和FWHM值

2. **Load Ground Spectra**: Read TSG8 spectrum files and validate data
   - 加载地面光谱：读取TSG8光谱文件并验证数据

3. **Apply Savitzky-Golay Filter**: Denoise spectra while preserving absorption features
   - 应用Savitzky-Golay滤波器：在保留吸收特征的同时对光谱进行去噪

4. **Resample to Satellite Bands**: Convert high-resolution ground spectra to satellite band resolution using Gaussian SRF convolution
   - 重采样到卫星波段：使用高斯SRF卷积将高分辨率地面光谱转换为卫星波段分辨率

5. **Aggregate by Class**: Compute mean reference spectrum for each lithology class
   - 按类别聚合：计算每个岩性类别的平均参考光谱

6. **Save and Visualize**: Save GSRSL to file and generate visualization
   - 保存和可视化：将GSRSL保存到文件并生成可视化

## Development
## 开发

### Running Tests
### 运行测试

Run all tests:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_metadata_parser.py -v
pytest tests/test_resampler.py -v
pytest tests/test_build_gsrsl.py -v
```

Run with coverage report:
```bash
pytest tests/ --cov=gsrsl_pipeline --cov-report=html --cov-report=term
```

### Test Coverage
### 测试覆盖率

The project includes comprehensive testing:
- **131 unit tests** covering all modules
- **Integration tests** for end-to-end pipeline validation
- **Property-based tests** using Hypothesis for mathematical correctness
- **Coverage**: >90% line coverage, >85% branch coverage

项目包含全面的测试：
- **131个单元测试**覆盖所有模块
- **集成测试**用于端到端管道验证
- **属性测试**使用Hypothesis验证数学正确性
- **覆盖率**：>90%行覆盖率，>85%分支覆盖率

### Code Structure
### 代码结构

The pipeline follows a modular architecture:

管道遵循模块化架构：

- `metadata_parser.py`: Parse GF-5 satellite metadata
- `data_models.py`: Data structures (GroundSpectrum)
- `spectrum_loader.py`: Load and validate ground spectra
- `filters.py`: Savitzky-Golay denoising
- `resampler.py`: Spectral resampling with Gaussian SRF
- `aggregator.py`: Class-wise spectrum aggregation
- `output.py`: Save GSRSL and generate visualizations
- `logging_config.py`: Logging configuration

## Troubleshooting
## 故障排除

### Common Issues
### 常见问题

**1. FileNotFoundError: Metadata file not found**
- Verify the metadata file path is correct
- Ensure the file exists and is readable
- 验证元数据文件路径是否正确
- 确保文件存在且可读

**2. ValueError: Expected exactly 297 bands**
- Check that the metadata file contains all 297 wavelength and FWHM entries
- Verify the file format matches the expected pattern
- 检查元数据文件是否包含所有297个波长和FWHM条目
- 验证文件格式是否符合预期模式

**3. KeyError: Filename not in label table**
- Ensure all spectrum filenames are listed in the label CSV
- Check for typos in filenames
- 确保所有光谱文件名都在标签CSV中列出
- 检查文件名中的拼写错误

**4. Warning: Reflectance values out of range**
- This is a warning, not an error - processing continues
- Values are automatically clipped to [0.0, 1.0] range
- Check your input data quality if many warnings appear
- 这是警告，不是错误 - 处理继续进行
- 值会自动裁剪到[0.0, 1.0]范围
- 如果出现许多警告，请检查输入数据质量

**5. Warning: Band has no overlapping ground data**
- Some satellite bands may not overlap with ground wavelength range
- These bands will have NaN values in the output
- This is expected for bands outside the 350-2500nm range
- 某些卫星波段可能与地面波长范围不重叠
- 这些波段在输出中将具有NaN值
- 对于350-2500nm范围之外的波段，这是预期的

### Performance Tips
### 性能提示

- **Large datasets**: Process spectra in batches if memory is limited
- **Parallel processing**: The pipeline is single-threaded; consider parallelizing spectrum loading for large datasets
- **Logging**: Use `--verbose` flag only for debugging; it generates large log files
- **大数据集**：如果内存有限，分批处理光谱
- **并行处理**：管道是单线程的；对于大数据集，考虑并行化光谱加载
- **日志记录**：仅在调试时使用`--verbose`标志；它会生成大型日志文件

## Scientific Background
## 科学背景

### Spectral Resampling Method
### 光谱重采样方法

The pipeline uses physically-accurate Gaussian spectral response function (SRF) convolution:

管道使用物理精确的高斯光谱响应函数(SRF)卷积：

1. **Gaussian SRF Model**: Each satellite band is modeled as a Gaussian function
   - 高斯SRF模型：每个卫星波段建模为高斯函数

2. **FWHM to Sigma Conversion**: σ = FWHM / (2√(2ln2)) ≈ FWHM / 2.355
   - FWHM到Sigma转换：σ = FWHM / (2√(2ln2)) ≈ FWHM / 2.355

3. **Weighted Integration**: R_sat = Σ(R_ground × w) / Σ(w)
   - 加权积分：R_sat = Σ(R_ground × w) / Σ(w)

4. **Integration Range**: ±3σ covers 99.7% of Gaussian distribution
   - 积分范围：±3σ覆盖99.7%的高斯分布

This method ensures spectral compatibility between ground measurements and satellite observations.

该方法确保地面测量和卫星观测之间的光谱兼容性。

## Citation
## 引用

If you use this pipeline in your research, please cite:

如果您在研究中使用此管道，请引用：

```
[Citation information to be added]
[引用信息待添加]
```

## License
## 许可证

TBD

## Contact
## 联系方式

TBD
