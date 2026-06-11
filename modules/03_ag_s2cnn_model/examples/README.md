# AG-S²CNN 使用示例

本目录包含 AG-S²CNN 模型的各种使用示例，帮助您快速上手。

## 示例列表

### 1. 数据预处理 (`demo_data_preprocessing.py`)

演示如何预处理高光谱影像数据，包括：
- 加载多种格式的影像（.npy, .tif, .mat）
- 数据归一化（最小-最大、Z-score、百分位数）
- 移除坏波段
- 创建 Ground Truth 标签图
- 创建地面标准参考光谱库 (GSRSL)
- 数据可视化

**使用方法:**

```bash
# 基本预处理
python examples/demo_data_preprocessing.py \
    --input data/raw_image.npy \
    --output data/processed_image.npy \
    --normalize minmax

# 完整预处理（包括 GT 和 GSRSL）
python examples/demo_data_preprocessing.py \
    --input data/raw_image.npy \
    --output data/processed_image.npy \
    --gt-output data/ground_truth.npy \
    --gsrsl-output data/gsrsl.npy \
    --num-classes 4 \
    --visualize

# 移除坏波段
python examples/demo_data_preprocessing.py \
    --input data/raw_image.npy \
    --output data/processed_image.npy \
    --bad-bands 0 1 2 100 101 102 \
    --normalize percentile
```

### 2. 模型训练 (`demo_training.py`)

演示完整的模型训练流程，包括：
- 加载配置和数据
- 创建模型和优化器
- 训练循环和验证
- 早停机制
- 模型保存
- 生成训练曲线和混淆矩阵

**使用方法:**

```bash
# 使用默认配置训练
python examples/demo_training.py

# 使用自定义配置
python examples/demo_training.py --config my_config.yaml
```

**在 Python 脚本中使用:**

```python
from examples.demo_training import train_model

# 训练模型
train_model(config_path='config.yaml')
```

### 3. 模型推理 (`demo_inference.py`)

演示如何使用训练好的模型进行推理，包括：
- 加载训练好的模型
- 单样本推理
- 全图推理
- 生成预测图和置信度图
- 结果可视化

**使用方法:**

```bash
# 单样本推理（使用模拟数据）
python examples/demo_inference.py \
    --config config.yaml \
    --model outputs/best_model.pth

# 全图推理
python examples/demo_inference.py \
    --config config.yaml \
    --model outputs/best_model.pth \
    --image data/test_image.npy \
    --gsrsl data/gsrsl.npy \
    --output results \
    --batch-size 32

# 使用 CPU 推理
python examples/demo_inference.py \
    --config config.yaml \
    --model outputs/best_model.pth \
    --image data/test_image.npy \
    --device cpu
```

**在 Python 脚本中使用:**

```python
from examples.demo_inference import predict_single_sample, predict_image
import numpy as np
import torch

# 加载模型
from models.ag_s2cnn import AG_S2CNN
model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
model.load_state_dict(torch.load('outputs/best_model.pth'))

# 单样本推理
x_sat = np.random.randn(290, 13, 13).astype(np.float32)
x_ref = np.random.randn(290).astype(np.float32)
prediction, probabilities = predict_single_sample(model, x_sat, x_ref)

print(f"预测类别: {prediction}")
print(f"类别概率: {probabilities}")
```

### 4. 损失和指标演示 (`demo_loss_metrics.py`)

演示损失函数和评估指标的使用，包括：
- 加权交叉熵损失
- 类别权重计算
- 评估指标计算（OA, AA, Kappa, F1-Score）
- 混淆矩阵生成

**使用方法:**

```bash
python examples/demo_loss_metrics.py
```

### 5. 日志和可视化演示 (`demo_logger_visualization.py`)

演示日志系统和可视化工具的使用，包括：
- 日志配置
- 训练曲线绘制
- 混淆矩阵可视化
- 学习率曲线

**使用方法:**

```bash
python examples/demo_logger_visualization.py
```

## 完整工作流程

### 步骤 1: 数据准备

```bash
# 预处理原始数据
python examples/demo_data_preprocessing.py \
    --input data/raw_image.npy \
    --output data/processed_image.npy \
    --gt-output data/ground_truth.npy \
    --gsrsl-output data/gsrsl.npy \
    --num-classes 4 \
    --normalize minmax \
    --visualize
```

### 步骤 2: 配置模型

编辑 `config.yaml` 文件，设置数据路径和超参数：

```yaml
model:
  num_bands: 290
  spatial_size: 13
  num_classes: 4

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100

data:
  image_path: 'data/processed_image.npy'
  ground_truth_path: 'data/ground_truth.npy'
  gsrsl_path: 'data/gsrsl.npy'
  output_dir: './outputs'
```

### 步骤 3: 训练模型

```bash
# 开始训练
python examples/demo_training.py --config config.yaml
```

训练过程中会自动：
- 保存最佳模型到 `outputs/best_model.pth`
- 生成训练曲线 `outputs/loss_curves.png`
- 生成混淆矩阵 `outputs/confusion_matrix.png`
- 记录日志到 `outputs/training.log`

### 步骤 4: 模型推理

```bash
# 对测试数据进行推理
python examples/demo_inference.py \
    --config config.yaml \
    --model outputs/best_model.pth \
    --image data/test_image.npy \
    --gsrsl data/gsrsl.npy \
    --output results
```

结果将保存到 `results/` 目录：
- `prediction_map.npy`: 预测结果图
- `confidence_map.npy`: 置信度图
- `prediction_result.png`: 可视化结果

## 自定义示例

### 创建自定义训练脚本

```python
import torch
from torch.utils.data import DataLoader
from models.ag_s2cnn import AG_S2CNN
from utils.dataset import HyperspectralDataset
from utils.loss import WeightedCrossEntropyLoss
from config import Config

# 加载配置
config = Config.from_yaml('config.yaml')

# 创建数据集
dataset = HyperspectralDataset(
    image_cube=image,
    ground_truth=gt,
    gsrsl=gsrsl,
    spatial_size=13,
    augmentation=True
)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 创建模型
model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 创建损失函数和优化器
criterion = WeightedCrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(100):
    model.train()
    for x_sat, x_ref, labels in dataloader:
        x_sat, x_ref, labels = x_sat.to(device), x_ref.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_sat, x_ref)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), 'my_model.pth')
```

### 创建自定义推理脚本

```python
import torch
import numpy as np
from models.ag_s2cnn import AG_S2CNN

# 加载模型
model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
model.load_state_dict(torch.load('my_model.pth'))
model.eval()

# 准备数据
x_sat = torch.randn(1, 1, 290, 13, 13)
x_ref = torch.randn(1, 1, 290, 1, 1)

# 推理
with torch.no_grad():
    output = model(x_sat, x_ref)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1)

print(f"预测类别: {prediction.item()}")
print(f"类别概率: {probabilities.numpy()}")
```

## 常见问题

### Q: 如何处理大型影像？

A: 对于大型影像，建议：
1. 使用较小的批量大小
2. 分块处理影像
3. 使用 `demo_inference.py` 的批量推理功能

### Q: 如何调整超参数？

A: 编辑 `config.yaml` 文件或创建新的配置文件：
- `batch_size`: 根据 GPU 内存调整（推荐 16-64）
- `learning_rate`: 学习率（推荐 1e-4 到 1e-3）
- `num_epochs`: 训练轮数（推荐 50-200）

### Q: 如何使用自己的数据？

A: 按照以下步骤：
1. 使用 `demo_data_preprocessing.py` 预处理数据
2. 确保数据格式正确：
   - 影像: (H, W, Bands) 的 NumPy 数组
   - Ground Truth: (H, W) 的整数数组
   - GSRSL: {class_id: spectrum} 的字典
3. 更新 `config.yaml` 中的路径

### Q: 训练时 GPU 内存不足怎么办？

A: 尝试以下方法：
1. 减小批量大小
2. 减小邻域大小（spatial_size）
3. 使用梯度累积
4. 使用混合精度训练

## 更多资源

- [API 文档](../docs/api_documentation.md)
- [配置使用指南](../docs/config_cli_usage.md)
- [训练流水线文档](../docs/training_pipeline_usage.md)
- [错误处理指南](../docs/error_handling_implementation.md)

## 贡献

欢迎贡献新的示例！请确保：
1. 代码清晰易懂
2. 包含详细的注释
3. 提供使用说明
4. 测试代码可以正常运行
