# GSRSL Pipeline Examples
# GSRSL管道示例

This directory contains example scripts and demo data for the GSRSL pipeline.

该目录包含GSRSL管道的示例脚本和演示数据。

## Quick Start
## 快速开始

### Run Complete Demo
### 运行完整演示

**Linux/Mac:**
```bash
bash examples/run_example.sh
```

**Windows:**
```cmd
examples\run_example.bat
```

**Or directly with Python:**
```bash
python examples/demo_build_gsrsl.py
```

This demo:
- Creates synthetic ground spectra for all 5 lithology classes
- Runs the complete GSRSL pipeline
- Verifies outputs
- Shows expected results

该演示：
- 为所有5个岩性类别创建合成地面光谱
- 运行完整的GSRSL管道
- 验证输出
- 显示预期结果

## Example Scripts
## 示例脚本

### 1. Complete Pipeline Demo
### 1. 完整管道演示

**File:** `demo_build_gsrsl.py`

Demonstrates the complete end-to-end pipeline with synthetic data:
- Creates 15 synthetic ground spectra (3 samples per class)
- Generates GF-5 metadata file
- Creates label table
- Runs the full pipeline
- Verifies outputs

演示使用合成数据的完整端到端管道：
- 创建15个合成地面光谱（每个类别3个样本）
- 生成GF-5元数据文件
- 创建标签表
- 运行完整管道
- 验证输出

**Usage:**
```bash
python examples/demo_build_gsrsl.py
```

### 2. Spectral Resampler Demo
### 2. 光谱重采样器演示

**File:** `demo_resampler.py`

Demonstrates the spectral resampling component:
- Shows how to use the SpectralResampler class
- Visualizes the resampling process
- Compares ground and satellite spectra

演示光谱重采样组件：
- 展示如何使用SpectralResampler类
- 可视化重采样过程
- 比较地面和卫星光谱

**Usage:**
```bash
python examples/demo_resampler.py
```

**Output:** `examples/resampler_demo_output.png`

### 3. Savitzky-Golay Filter Demo
### 3. Savitzky-Golay滤波器演示

**File:** `demo_savgol_filter.py`

Demonstrates the Savitzky-Golay denoising filter:
- Shows before/after filtering
- Visualizes noise reduction
- Preserves absorption features

演示Savitzky-Golay去噪滤波器：
- 显示滤波前后对比
- 可视化噪声降低
- 保留吸收特征

**Usage:**
```bash
python examples/demo_savgol_filter.py
```

**Output:** `examples/savgol_filter_demo.png`

### 4. Class Aggregator Demo
### 4. 类别聚合器演示

**File:** `demo_aggregator.py`

Demonstrates class-wise spectrum aggregation:
- Shows how to aggregate multiple spectra per class
- Handles NaN values correctly
- Computes mean reference spectra

演示按类别聚合光谱：
- 展示如何聚合每个类别的多个光谱
- 正确处理NaN值
- 计算平均参考光谱

**Usage:**
```bash
python examples/demo_aggregator.py
```

### 5. Output Generator Demo
### 5. 输出生成器演示

**File:** `demo_output.py`

Demonstrates saving and visualizing GSRSL:
- Shows how to save GSRSL to file
- Generates visualization plots
- Demonstrates loading saved GSRSL

演示保存和可视化GSRSL：
- 展示如何将GSRSL保存到文件
- 生成可视化图
- 演示加载保存的GSRSL

**Usage:**
```bash
python examples/demo_output.py
```

**Outputs:**
- `examples/demo_gsrsl.npy`
- `examples/demo_gsrsl_visualization.png`
- `examples/demo_gsrsl_with_nan.npy`
- `examples/demo_gsrsl_with_nan_visualization.png`

## Demo Outputs
## 演示输出

After running the demos, you'll find the following output files:

运行演示后，您将找到以下输出文件：

- **gsrsl.npy**: NumPy dictionary with reference spectra for all 5 classes
  - NumPy字典，包含所有5个类别的参考光谱
  
- **gsrsl_visualization.png**: Plot showing all 5 reference spectra
  - 显示所有5个参考光谱的图

- **demo.log**: Detailed log file with processing information
  - 包含处理信息的详细日志文件

## Understanding the Output
## 理解输出

### GSRSL File Structure
### GSRSL文件结构

The `gsrsl.npy` file contains a Python dictionary:

`gsrsl.npy`文件包含一个Python字典：

```python
{
    0: array([...], shape=(297,), dtype=float32),  # Spodumene-rich
    1: array([...], shape=(297,), dtype=float32),  # Lepidolite-rich
    2: array([...], shape=(297,), dtype=float32),  # Mixed-type
    3: array([...], shape=(297,), dtype=float32),  # Barren
    4: array([...], shape=(297,), dtype=float32)   # Wall Rock
}
```

Each array contains 297 reflectance values corresponding to GF-5 satellite bands.

每个数组包含297个反射率值，对应于GF-5卫星波段。

### Loading GSRSL
### 加载GSRSL

```python
import numpy as np

# Load GSRSL
gsrsl = np.load('gsrsl.npy', allow_pickle=True).item()

# Access specific class
spodumene_spectrum = gsrsl[0]

# Check shape and dtype
print(spodumene_spectrum.shape)  # (297,)
print(spodumene_spectrum.dtype)  # float32

# Get valid (non-NaN) bands
valid_mask = ~np.isnan(spodumene_spectrum)
valid_reflectances = spodumene_spectrum[valid_mask]
```

## Lithology Classes
## 岩性类别

The pipeline processes 5 lithology classes:

管道处理5个岩性类别：

| Class ID | English Name | Chinese Name | Description |
|----------|--------------|--------------|-------------|
| 0 | Spodumene-rich Pegmatite | 锂辉石型富矿伟晶岩 | Rich in spodumene (LiAlSi₂O₆) |
| 1 | Lepidolite-rich Pegmatite | 锂云母型富矿伟晶岩 | Rich in lepidolite (K(Li,Al)₃(Si,Al)₄O₁₀(F,OH)₂) |
| 2 | Mixed-type Rich Pegmatite | 混合型富矿伟晶岩 | Mixed lithium minerals |
| 3 | Barren Pegmatite | 贫矿伟晶岩 | Low lithium content |
| 4 | Wall Rock | 围岩 | Surrounding host rock |

## Customizing Examples
## 自定义示例

You can modify the demo scripts to:
- Change the number of samples per class
- Adjust spectral characteristics
- Test different parameter settings
- Add your own data

您可以修改演示脚本以：
- 更改每个类别的样本数量
- 调整光谱特征
- 测试不同的参数设置
- 添加您自己的数据

## Troubleshooting
## 故障排除

**Issue:** Demo fails with "Module not found"
- **Solution:** Install dependencies: `pip install -r requirements.txt`

**Issue:** Visualization not displayed
- **Solution:** Check that matplotlib backend is configured correctly

**Issue:** Permission denied on shell scripts
- **Solution:** Make scripts executable: `chmod +x examples/run_example.sh`

**问题：** 演示失败，提示"未找到模块"
- **解决方案：** 安装依赖项：`pip install -r requirements.txt`

**问题：** 可视化未显示
- **解决方案：** 检查matplotlib后端是否正确配置

**问题：** shell脚本权限被拒绝
- **解决方案：** 使脚本可执行：`chmod +x examples/run_example.sh`

## Next Steps
## 下一步

After running the examples:

运行示例后：

1. Review the generated visualizations
   - 查看生成的可视化图

2. Inspect the log files to understand the processing steps
   - 检查日志文件以了解处理步骤

3. Try running the pipeline with your own data
   - 尝试使用您自己的数据运行管道

4. Explore the individual component demos to understand each step
   - 探索各个组件演示以了解每个步骤

## Additional Resources
## 其他资源

- Main README: `../README.md`
- Design Document: `../.kiro/specs/gsrsl-pipeline/design.md`
- Requirements: `../.kiro/specs/gsrsl-pipeline/requirements.md`
- Test Suite: `../tests/`

## Contact
## 联系方式

For questions or issues with the examples, please refer to the main project documentation.

有关示例的问题或疑问，请参阅主项目文档。
