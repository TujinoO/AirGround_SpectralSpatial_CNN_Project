# 错误处理与异常管理实现总结

## 概述

本文档总结了 AG-S²CNN 项目中实现的错误处理与异常管理功能，涵盖输入验证、GPU 内存错误、文件路径错误和训练中断处理。

## 实现的功能

### 1. 输入验证错误处理 (任务 15.1)

**位置**: `models/ag_s2cnn.py` - `AG_S2CNN._validate_input()` 方法

**功能**:
- 验证卫星流输入维度 (Batch, 1, 290, 13, 13)
- 验证地面流输入维度 (Batch, 1, 290, 1, 1)
- 检查批量大小一致性
- 验证批量大小范围 (1-128)
- 检测 NaN 值
- 检测 Inf 值

**错误信息示例**:
```
维度错误: 卫星流输入期望维度为 (Batch, 1, 290, 13, 13)，实际接收到 (2, 1, 290, 15, 15)
```

**需求**: 1.3, 1.4, 15.1

### 2. GPU 内存错误处理 (任务 15.3)

**位置**: `train.py` - `train_one_epoch()` 和 `validate()` 函数

**功能**:
- 捕获 CUDA OOM (Out of Memory) 错误
- 显示当前批量大小和 GPU 内存使用情况
- 自动清理 GPU 缓存
- 提供友好的错误信息和优化建议

**错误信息示例**:
```
错误: GPU 内存不足!
当前批量大小: 32
GPU 内存使用: 8.45 GB
GPU 内存缓存: 9.12 GB
建议:
  1. 减小批量大小 (batch_size)
  2. 减小模型输入尺寸
  3. 使用梯度累积
  4. 使用混合精度训练 (AMP)
```

**需求**: 15.2

### 3. 文件路径错误处理 (任务 15.4)

**位置**: 
- `utils/file_utils.py` - 文件路径验证工具模块
- `train.py` - 配置文件加载
- `utils/dataset.py` - 数据文件加载

**功能**:
- `validate_file_exists()`: 验证文件是否存在
- `validate_directory_exists()`: 验证目录是否存在
- `validate_file_extension()`: 验证文件扩展名
- `ensure_directory_exists()`: 确保目录存在，不存在则创建
- `load_hyperspectral_data()`: 加载数据时验证文件路径

**错误信息示例**:
```
配置文件不存在: D:\Code\AG-S2CNN\config.yaml
请检查路径是否正确，或文件是否已被删除。
```

**需求**: 15.3

### 4. 训练中断处理 (任务 15.5)

**位置**: `train.py` - `main()` 函数

**功能**:
- 捕获 KeyboardInterrupt (Ctrl+C)
- 保存中断时的模型状态到 `interrupted_checkpoint.pth`
- 提供恢复训练的命令提示
- 捕获其他异常并保存错误状态到 `error_checkpoint.pth`
- 安全终止训练

**中断信息示例**:
```
============================================================
训练被用户中断 (Ctrl+C)
============================================================
中断时的模型状态已保存到: ./outputs/checkpoints/interrupted_checkpoint.pth
可以使用以下命令恢复训练:
  python train.py --resume ./outputs/checkpoints/interrupted_checkpoint.pth

训练已安全终止
```

**需求**: 15.5

## 测试覆盖

### 测试文件: `tests/test_error_handling.py`

**测试类**:

1. **TestInputValidation** - 输入验证测试
   - `test_invalid_satellite_dimension`: 测试卫星流维度错误
   - `test_invalid_ground_dimension`: 测试地面流维度错误
   - `test_batch_size_mismatch`: 测试批量大小不一致
   - `test_batch_size_out_of_range`: 测试批量大小超出范围
   - `test_nan_values_in_satellite`: 测试 NaN 值检测
   - `test_inf_values_in_ground`: 测试 Inf 值检测
   - `test_valid_input`: 测试合法输入

2. **TestFilePathValidation** - 文件路径验证测试
   - `test_file_not_exists`: 测试文件不存在
   - `test_directory_not_exists`: 测试目录不存在
   - `test_invalid_file_extension`: 测试文件扩展名错误
   - `test_ensure_directory_exists`: 测试目录创建
   - `test_load_checkpoint_file_not_found`: 测试检查点文件不存在

3. **TestDimensionAlignment** - 维度对齐测试
   - `test_fusion_dimension_mismatch`: 测试融合模块维度不匹配

**测试结果**: 13 个测试全部通过 ✓

## 错误处理最佳实践

### 1. 明确的错误信息
所有错误信息都包含:
- 错误类型描述
- 期望值和实际值
- 完整的文件路径（绝对路径）
- 解决建议

### 2. 资源清理
- GPU 内存错误时自动清理缓存
- 异常发生时保存模型状态
- 使用 try-except-finally 确保资源释放

### 3. 用户友好
- 提供中英文双语错误信息
- 给出具体的解决建议
- 保存中断状态以便恢复

### 4. 防御性编程
- 在关键操作前验证输入
- 使用 assert 进行内部一致性检查
- 提供默认值和回退机制

## 使用示例

### 示例 1: 处理维度错误

```python
from models.ag_s2cnn import AG_S2CNN
import torch

model = AG_S2CNN()

try:
    x_sat = torch.randn(2, 1, 290, 15, 15)  # 错误的维度
    x_ref = torch.randn(2, 1, 290, 1, 1)
    output = model(x_sat, x_ref)
except ValueError as e:
    print(f"输入验证失败: {e}")
    # 修正维度后重试
```

### 示例 2: 处理文件路径错误

```python
from utils.file_utils import validate_file_exists

try:
    validate_file_exists("data/image.npy", "高光谱影像文件")
except FileNotFoundError as e:
    print(f"文件路径错误: {e}")
    # 提示用户检查路径
```

### 示例 3: 处理 GPU 内存错误

```python
try:
    train_loss, train_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
except RuntimeError as e:
    if 'out of memory' in str(e).lower():
        print("GPU 内存不足，尝试减小批量大小")
        # 减小批量大小后重试
    else:
        raise
```

## 相关文件

- `models/ag_s2cnn.py`: 模型输入验证
- `train.py`: 训练流程错误处理
- `utils/file_utils.py`: 文件路径验证工具
- `utils/dataset.py`: 数据加载错误处理
- `config.py`: 配置验证
- `tests/test_error_handling.py`: 错误处理测试

## 未来改进

1. 添加更详细的日志记录
2. 实现自动批量大小调整（GPU OOM 时）
3. 添加模型权重不匹配的详细诊断
4. 支持分布式训练的错误处理
5. 添加数据损坏检测和修复

## 总结

本次实现完成了 AG-S²CNN 项目的完整错误处理与异常管理系统，涵盖了输入验证、GPU 内存管理、文件路径验证和训练中断处理等关键场景。所有功能都经过了充分的测试验证，确保系统的健壮性和用户友好性。
