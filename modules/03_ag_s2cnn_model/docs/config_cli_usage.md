# 配置管理与命令行接口使用指南

## 概述

AG-S²CNN 项目提供了完整的配置管理系统和命令行接口，支持灵活的参数配置和多种运行模式。

## 配置管理

### 配置类 (Config)

配置类位于 `config.py`，提供以下功能：

- 从 YAML 文件加载配置
- 验证配置完整性和合法性
- 提供默认值
- 支持嵌套键访问（点号分隔）
- 保存配置到文件

### 使用示例

```python
from config import Config

# 1. 使用默认配置
config = Config()

# 2. 从 YAML 文件加载
config = Config.from_yaml('config.yaml')

# 3. 访问配置项（点号分隔）
num_bands = config.get('model.num_bands')
batch_size = config.get('training.batch_size')

# 4. 访问配置项（字典式）
num_bands = config['model']['num_bands']

# 5. 保存配置
config.save('output_config.yaml')
```

### 配置文件结构

配置文件 `config.yaml` 包含以下部分：

```yaml
# 模型配置
model:
  num_bands: 290          # 光谱波段数
  spatial_size: 13        # 空间邻域大小（必须为奇数）
  num_classes: 4          # 分类类别数

# 训练配置
training:
  batch_size: 32          # 批量大小
  learning_rate: 0.001    # 学习率
  weight_decay: 0.0001    # 权重衰减
  num_epochs: 100         # 训练轮数
  early_stopping_patience: 20  # 早停耐心值

# 数据配置
data:
  image_path: ""          # 高光谱影像路径
  ground_truth_path: ""   # Ground Truth 标签图路径
  gsrsl_path: ""          # 地面标准参考光谱库路径
  output_dir: "./outputs" # 输出目录

# 数据增强配置
augmentation:
  enabled: true           # 是否启用数据增强
  noise_std: 0.01         # 光谱噪声标准差

# 设备配置
device:
  use_cuda: true          # 是否使用 GPU
  gpu_id: 0               # GPU 设备 ID
```

## 命令行接口

### 基本用法

```bash
# 显示帮助信息
python train.py --help

# 使用默认配置训练
python train.py

# 使用自定义配置文件
python train.py --config my_config.yaml
```

### 运行模式

支持三种运行模式：

1. **训练模式** (默认)
```bash
python train.py --mode train
```

2. **验证模式**
```bash
python train.py --mode validate --checkpoint outputs/checkpoints/best_model.pth
```

3. **测试模式**
```bash
python train.py --mode test --checkpoint outputs/checkpoints/best_model.pth
```

### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--config` | `-c` | 配置文件路径 | `config.yaml` |
| `--mode` | `-m` | 运行模式 (train/validate/test) | `train` |
| `--resume` | `-r` | 从检查点恢复训练的路径 | `None` |
| `--checkpoint` | - | 验证/测试用的检查点路径 | `None` |
| `--gpu` | - | GPU 设备 ID | 配置文件中的值 |
| `--batch-size` | - | 批量大小 | 配置文件中的值 |
| `--epochs` | - | 训练轮数 | 配置文件中的值 |
| `--lr` | - | 学习率 | 配置文件中的值 |
| `--output-dir` | - | 输出目录 | 配置文件中的值 |

### 参数覆盖

命令行参数会覆盖配置文件中的设置：

```bash
# 覆盖批量大小和学习率
python train.py --batch-size 64 --lr 0.0001

# 覆盖 GPU 设备
python train.py --gpu 1

# 覆盖输出目录
python train.py --output-dir ./my_outputs
```

### 从检查点恢复训练

```bash
# 从指定检查点恢复训练
python train.py --resume outputs/checkpoints/checkpoint_epoch_50.pth

# 恢复训练并修改学习率
python train.py --resume outputs/checkpoints/checkpoint_epoch_50.pth --lr 0.0001
```

## 使用场景示例

### 场景 1: 快速开始训练

```bash
# 使用默认配置开始训练
python train.py
```

### 场景 2: 自定义配置训练

```bash
# 1. 复制并修改配置文件
cp config.yaml my_config.yaml
# 2. 编辑 my_config.yaml
# 3. 使用自定义配置训练
python train.py --config my_config.yaml
```

### 场景 3: 超参数调优

```bash
# 测试不同的批量大小
python train.py --batch-size 16 --output-dir ./outputs/bs16
python train.py --batch-size 32 --output-dir ./outputs/bs32
python train.py --batch-size 64 --output-dir ./outputs/bs64

# 测试不同的学习率
python train.py --lr 0.001 --output-dir ./outputs/lr001
python train.py --lr 0.0001 --output-dir ./outputs/lr0001
```

### 场景 4: 模型评估

```bash
# 验证最佳模型
python train.py --mode validate --checkpoint outputs/checkpoints/best_model.pth

# 测试模型
python train.py --mode test --checkpoint outputs/checkpoints/best_model.pth
```

### 场景 5: 训练中断后恢复

```bash
# 训练过程中按 Ctrl+C 中断
# 系统会自动保存 interrupted_checkpoint.pth

# 从中断点恢复训练
python train.py --resume outputs/checkpoints/interrupted_checkpoint.pth
```

## 配置验证

配置类会自动验证以下内容：

- ✓ 波段数必须为正整数
- ✓ 邻域大小必须为正奇数
- ✓ 类别数必须为正整数
- ✓ 批量大小必须为正整数
- ✓ 学习率必须为正数
- ✓ 训练轮数必须为正整数
- ✓ 输出目录存在性（不存在会自动创建）

## 错误处理

### 配置文件不存在

```bash
$ python train.py --config nonexistent.yaml
警告: 配置文件 nonexistent.yaml 不存在，使用默认配置
```

### 验证模式缺少检查点

```bash
$ python train.py --mode validate
错误: 验证模式需要指定 --checkpoint 参数
```

### 非法配置值

```python
# 邻域大小为偶数
Config({'model': {'spatial_size': 12}})
# ValueError: 邻域大小必须为正奇数，当前值: 12
```

## 最佳实践

1. **使用配置文件管理实验**
   - 为每个实验创建独立的配置文件
   - 使用有意义的文件名（如 `config_baseline.yaml`, `config_augmented.yaml`）

2. **保存配置到输出目录**
   - 训练脚本会自动保存使用的配置到 `outputs/config_used.yaml`
   - 便于追踪实验设置

3. **使用命令行参数进行快速调试**
   - 小规模测试时使用 `--batch-size 8 --epochs 5`
   - 避免修改配置文件

4. **利用早停机制**
   - 设置合理的 `early_stopping_patience` 值
   - 避免过拟合和浪费计算资源

5. **定期保存检查点**
   - 训练脚本会自动保存最佳模型和定期检查点
   - 使用 `--resume` 从检查点恢复训练

## 相关文件

- `config.py` - 配置管理类实现
- `config.yaml` - 示例配置文件
- `train.py` - 训练脚本（包含命令行接口）
- `tests/test_config.py` - 配置管理单元测试

## 需求追溯

本实现满足以下需求：

- **需求 13.1-13.6**: 配置管理功能
- **任务 16.1**: 实现配置类
- **任务 16.3**: 创建示例配置文件
- **任务 16.4**: 实现命令行接口
