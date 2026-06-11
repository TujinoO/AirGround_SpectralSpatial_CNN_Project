# AG-S²CNN API 文档

本文档详细描述了 AG-S²CNN 模型的所有公共类和方法的 API 接口。

## 目录

- [模型模块 (models)](#模型模块-models)
  - [AG_S2CNN](#ag_s2cnn)
  - [GEncoder](#gencoder)
  - [SBackbone](#sbackbone)
  - [AGDifferenceFusion](#agdifferencefusion)
  - [Classifier2D](#classifier2d)
  - [ResNetBlock2D](#resnetblock2d)
  - [Inception3D](#inception3d)
- [工具模块 (utils)](#工具模块-utils)
  - [HyperspectralDataset](#hyperspectraldataset)
  - [DataAugmentation](#dataaugmentation)
  - [WeightedCrossEntropyLoss](#weightedcrossentropyloss)
  - [MetricsCalculator](#metricscalculator)
  - [Config](#config)
- [训练模块](#训练模块)
  - [train_one_epoch](#train_one_epoch)
  - [validate](#validate)
  - [EarlyStopping](#earlystopping)

---

## 模型模块 (models)

### AG_S2CNN

```python
class AG_S2CNN(nn.Module)
```

AG-S²CNN 主模型类，空地协同光谱-空间卷积神经网络。

**参数:**
- `num_bands` (int, 可选): 光谱波段数，默认 290
- `spatial_size` (int, 可选): 空间邻域大小，默认 13
- `num_classes` (int, 可选): 分类类别数，默认 4

**属性:**
- `num_bands` (int): 光谱波段数
- `spatial_size` (int): 空间邻域大小
- `num_classes` (int): 分类类别数
- `reduced_bands` (int): 降维后的光谱维度
- `g_encoder` (GEncoder): 地面光谱编码器
- `s_backbone` (SBackbone): 卫星特征主干
- `ag_fusion` (AGDifferenceFusion): 差分融合模块
- `classifier` (Classifier2D): 分类器

**方法:**

#### `forward(x_sat, x_ref)`

前向传播方法。

**参数:**
- `x_sat` (torch.Tensor): 卫星流输入，形状 (Batch, 1, Bands, 13, 13)
- `x_ref` (torch.Tensor): 地面流输入，形状 (Batch, 1, Bands, 1, 1)

**返回:**
- `torch.Tensor`: 类别概率分布，形状 (Batch, num_classes)

**异常:**
- `ValueError`: 当输入维度不匹配或包含非法值时

**示例:**
```python
model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)
x_sat = torch.randn(32, 1, 290, 13, 13)
x_ref = torch.randn(32, 1, 290, 1, 1)
output = model(x_sat, x_ref)  # 形状: (32, 4)
```

---

### GEncoder

```python
class GEncoder(nn.Module)
```

地面光谱特征编码器，提取地面光谱的深层语义特征并执行空间广播。

**参数:**
- `num_bands` (int, 可选): 光谱波段数，默认 290
- `spatial_size` (int, 可选): 空间邻域大小，默认 13

**方法:**

#### `forward(x_ref)`

前向传播方法。

**参数:**
- `x_ref` (torch.Tensor): 地面流输入，形状 (Batch, 1, Bands, 1, 1)

**返回:**
- `torch.Tensor`: 广播后的特征，形状 (Batch, 64, D_new, S, S)

**示例:**
```python
encoder = GEncoder(num_bands=290, spatial_size=13)
x_ref = torch.randn(32, 1, 290, 1, 1)
f_ref = encoder(x_ref)  # 形状: (32, 64, 145, 13, 13)
```

---

### SBackbone

```python
class SBackbone(nn.Module)
```

多尺度卫星特征提取主干，使用 Inception-3D 结构提取多尺度特征。

**参数:**
- `num_bands` (int, 可选): 光谱波段数，默认 290
- `spatial_size` (int, 可选): 空间邻域大小，默认 13

**方法:**

#### `forward(x_sat)`

前向传播方法。

**参数:**
- `x_sat` (torch.Tensor): 卫星流输入，形状 (Batch, 1, Bands, 13, 13)

**返回:**
- `torch.Tensor`: 特征输出，形状 (Batch, 64, D_new, 13, 13)

**示例:**
```python
backbone = SBackbone(num_bands=290, spatial_size=13)
x_sat = torch.randn(32, 1, 290, 13, 13)
f_sat = backbone(x_sat)  # 形状: (32, 64, 145, 13, 13)
```

---

### AGDifferenceFusion

```python
class AGDifferenceFusion(nn.Module)
```

空地光谱差分融合模块，计算观测特征与标准特征的差异并融合。

**方法:**

#### `forward(f_sat, f_ref)`

前向传播方法。

**参数:**
- `f_sat` (torch.Tensor): 卫星特征，形状 (Batch, 64, D, H, W)
- `f_ref` (torch.Tensor): 地面参考特征，形状 (Batch, 64, D, H, W)

**返回:**
- `torch.Tensor`: 融合特征，形状 (Batch, 64, D, H, W)

**异常:**
- `AssertionError`: 当输入特征维度不一致时

**示例:**
```python
fusion = AGDifferenceFusion()
f_sat = torch.randn(32, 64, 145, 13, 13)
f_ref = torch.randn(32, 64, 145, 13, 13)
f_fused = fusion(f_sat, f_ref)  # 形状: (32, 64, 145, 13, 13)
```

---

### Classifier2D

```python
class Classifier2D(nn.Module)
```

2D 语义抽象与分类模块，将 3D 特征转换为类别概率。

**参数:**
- `reduced_bands` (int): 降维后的光谱维度
- `spatial_size` (int, 可选): 空间邻域大小，默认 13
- `num_classes` (int, 可选): 分类类别数，默认 4

**方法:**

#### `forward(x)`

前向传播方法。

**参数:**
- `x` (torch.Tensor): 3D 融合特征，形状 (Batch, 64, D, H, W)

**返回:**
- `torch.Tensor`: 类别概率，形状 (Batch, num_classes)

**示例:**
```python
classifier = Classifier2D(reduced_bands=145, spatial_size=13, num_classes=4)
x = torch.randn(32, 64, 145, 13, 13)
output = classifier(x)  # 形状: (32, 4)
```

---

### ResNetBlock2D

```python
class ResNetBlock2D(nn.Module)
```

2D 残差卷积块，包含两层 3×3 卷积和残差连接。

**参数:**
- `channels` (int): 输入和输出通道数

**方法:**

#### `forward(x)`

前向传播方法。

**参数:**
- `x` (torch.Tensor): 输入特征，形状 (Batch, Channels, H, W)

**返回:**
- `torch.Tensor`: 输出特征，形状 (Batch, Channels, H, W)

**示例:**
```python
block = ResNetBlock2D(channels=256)
x = torch.randn(32, 256, 13, 13)
out = block(x)  # 形状: (32, 256, 13, 13)
```

---

### Inception3D

```python
class Inception3D(nn.Module)
```

Inception-3D 多尺度并行模块，同时提取局部和全局特征。

**参数:**
- `in_channels` (int): 输入通道数

**方法:**

#### `forward(x)`

前向传播方法。

**参数:**
- `x` (torch.Tensor): 输入特征，形状 (Batch, in_channels, D, H, W)

**返回:**
- `torch.Tensor`: 输出特征，形状 (Batch, 48, D, H, W)

**示例:**
```python
inception = Inception3D(in_channels=16)
x = torch.randn(32, 16, 145, 13, 13)
out = inception(x)  # 形状: (32, 48, 145, 13, 13)
```

---

## 工具模块 (utils)

### HyperspectralDataset

```python
class HyperspectralDataset(Dataset)
```

高光谱数据集类，用于加载和处理高光谱影像数据。

**参数:**
- `image_cube` (numpy.ndarray): 全景高光谱影像，形状 (H, W, Bands)
- `ground_truth` (numpy.ndarray): 标签图，形状 (H, W)
- `gsrsl` (dict): 地面标准参考光谱库，格式 {class_id: spectrum}
- `spatial_size` (int, 可选): 邻域大小，默认 13
- `augmentation` (bool, 可选): 是否启用数据增强，默认 False

**方法:**

#### `__len__()`

返回数据集大小。

**返回:**
- `int`: 样本数量

#### `__getitem__(idx)`

获取单个样本。

**参数:**
- `idx` (int): 样本索引

**返回:**
- `tuple`: (x_sat, x_ref, label)
  - `x_sat` (torch.Tensor): 卫星流，形状 (1, Bands, S, S)
  - `x_ref` (torch.Tensor): 地面流，形状 (1, Bands, 1, 1)
  - `label` (int): 类别标签

**示例:**
```python
dataset = HyperspectralDataset(
    image_cube=image,
    ground_truth=gt,
    gsrsl=gsrsl_dict,
    spatial_size=13,
    augmentation=True
)
x_sat, x_ref, label = dataset[0]
```

---

### DataAugmentation

```python
class DataAugmentation
```

数据增强类，提供几何变换和光谱噪声注入。

**参数:**
- `noise_std` (float, 可选): 噪声标准差，默认 0.01

**方法:**

#### `__call__(x_sat, x_ref)`

应用数据增强。

**参数:**
- `x_sat` (torch.Tensor): 卫星流，形状 (1, Bands, S, S)
- `x_ref` (torch.Tensor): 地面流，形状 (1, Bands, 1, 1)

**返回:**
- `tuple`: (x_sat_aug, x_ref)
  - `x_sat_aug` (torch.Tensor): 增强后的卫星流
  - `x_ref` (torch.Tensor): 未修改的地面流

**示例:**
```python
augmenter = DataAugmentation(noise_std=0.01)
x_sat_aug, x_ref = augmenter(x_sat, x_ref)
```

---

### WeightedCrossEntropyLoss

```python
class WeightedCrossEntropyLoss(nn.Module)
```

加权交叉熵损失函数，用于处理类别不平衡问题。

**参数:**
- `class_weights` (torch.Tensor, 可选): 类别权重，形状 (num_classes,)

**方法:**

#### `forward(predictions, targets)`

计算损失。

**参数:**
- `predictions` (torch.Tensor): 模型输出，形状 (Batch, num_classes)
- `targets` (torch.Tensor): 真实标签，形状 (Batch,)

**返回:**
- `torch.Tensor`: 标量损失值

#### `calculate_class_weights(labels, num_classes)` (静态方法)

根据训练集计算类别权重。

**参数:**
- `labels` (numpy.ndarray): 所有训练标签
- `num_classes` (int): 类别数

**返回:**
- `torch.Tensor`: 类别权重

**示例:**
```python
# 计算权重
weights = WeightedCrossEntropyLoss.calculate_class_weights(train_labels, num_classes=4)

# 创建损失函数
criterion = WeightedCrossEntropyLoss(class_weights=weights)

# 计算损失
loss = criterion(predictions, targets)
```

---

### MetricsCalculator

```python
class MetricsCalculator
```

评估指标计算器，计算 OA、AA、Kappa、F1-Score 等指标。

**参数:**
- `num_classes` (int): 类别数

**方法:**

#### `calculate_all_metrics(y_true, y_pred)`

计算所有评估指标。

**参数:**
- `y_true` (numpy.ndarray): 真实标签
- `y_pred` (numpy.ndarray): 预测标签

**返回:**
- `dict`: 包含所有指标的字典
  - `'OA'` (float): Overall Accuracy
  - `'AA'` (float): Average Accuracy
  - `'Kappa'` (float): Kappa 系数
  - `'F1_per_class'` (numpy.ndarray): 每个类别的 F1-Score
  - `'F1_macro'` (float): 宏平均 F1-Score
  - `'Precision_per_class'` (numpy.ndarray): 每个类别的精确率
  - `'Recall_per_class'` (numpy.ndarray): 每个类别的召回率
  - `'confusion_matrix'` (numpy.ndarray): 混淆矩阵

#### `print_metrics(metrics, class_names=None)`

打印格式化的指标报告。

**参数:**
- `metrics` (dict): 指标字典
- `class_names` (list, 可选): 类别名称列表

**示例:**
```python
calculator = MetricsCalculator(num_classes=4)
metrics = calculator.calculate_all_metrics(y_true, y_pred)
calculator.print_metrics(metrics, class_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
```

---

### Config

```python
class Config
```

配置管理类，用于加载、验证和保存配置。

**类方法:**

#### `from_yaml(yaml_path)`

从 YAML 文件加载配置。

**参数:**
- `yaml_path` (str): YAML 文件路径

**返回:**
- `Config`: 配置对象

**异常:**
- `FileNotFoundError`: 当文件不存在时

**方法:**

#### `get(key, default=None)`

获取配置项（支持点号访问）。

**参数:**
- `key` (str): 配置键，支持 'model.num_bands' 格式
- `default` (Any, 可选): 默认值

**返回:**
- `Any`: 配置值

#### `validate()`

验证配置完整性。

**异常:**
- `ValueError`: 当配置不完整或非法时

#### `save(yaml_path)`

保存配置到 YAML 文件。

**参数:**
- `yaml_path` (str): 输出文件路径

**示例:**
```python
# 加载配置
config = Config.from_yaml('config.yaml')

# 访问配置
num_bands = config.get('model.num_bands')
batch_size = config['training']['batch_size']

# 验证配置
config.validate()

# 保存配置
config.save('output_config.yaml')
```

---

## 训练模块

### train_one_epoch

```python
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch)
```

训练一个 epoch。

**参数:**
- `model` (nn.Module): 模型
- `dataloader` (DataLoader): 训练数据加载器
- `criterion` (nn.Module): 损失函数
- `optimizer` (torch.optim.Optimizer): 优化器
- `device` (torch.device): 设备 (CPU/GPU)
- `epoch` (int): 当前 epoch 数

**返回:**
- `tuple`: (avg_loss, metrics)
  - `avg_loss` (float): 平均损失
  - `metrics` (dict): 训练指标字典

**示例:**
```python
avg_loss, metrics = train_one_epoch(
    model=model,
    dataloader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epoch=1
)
```

---

### validate

```python
def validate(model, dataloader, criterion, device, metrics_calculator)
```

验证模型。

**参数:**
- `model` (nn.Module): 模型
- `dataloader` (DataLoader): 验证数据加载器
- `criterion` (nn.Module): 损失函数
- `device` (torch.device): 设备 (CPU/GPU)
- `metrics_calculator` (MetricsCalculator): 指标计算器

**返回:**
- `tuple`: (avg_loss, metrics)
  - `avg_loss` (float): 平均损失
  - `metrics` (dict): 验证指标字典

**示例:**
```python
val_loss, val_metrics = validate(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    device=device,
    metrics_calculator=calculator
)
```

---

### EarlyStopping

```python
class EarlyStopping
```

早停机制类，监控验证指标并在未改善时停止训练。

**参数:**
- `patience` (int, 可选): 容忍的 epoch 数，默认 20
- `mode` (str, 可选): 监控模式，'min' 或 'max'，默认 'max'
- `delta` (float, 可选): 最小改善量，默认 0.0

**方法:**

#### `__call__(metric, model, save_path)`

检查是否应该早停。

**参数:**
- `metric` (float): 当前指标值
- `model` (nn.Module): 模型
- `save_path` (str): 模型保存路径

**返回:**
- `bool`: 是否应该停止训练

**示例:**
```python
early_stopping = EarlyStopping(patience=20, mode='max')

for epoch in range(num_epochs):
    # 训练和验证
    val_acc = validate(...)
    
    # 检查早停
    if early_stopping(val_acc, model, 'best_model.pth'):
        print("早停触发，停止训练")
        break
```

---

## 使用示例

### 完整训练流程

```python
import torch
from torch.utils.data import DataLoader
from models.ag_s2cnn import AG_S2CNN
from utils.dataset import HyperspectralDataset
from utils.loss import WeightedCrossEntropyLoss
from utils.metrics import MetricsCalculator
from config import Config

# 加载配置
config = Config.from_yaml('config.yaml')

# 创建模型
model = AG_S2CNN(
    num_bands=config['model']['num_bands'],
    spatial_size=config['model']['spatial_size'],
    num_classes=config['model']['num_classes']
)

# 创建数据集
train_dataset = HyperspectralDataset(
    image_cube=train_image,
    ground_truth=train_gt,
    gsrsl=gsrsl_dict,
    augmentation=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True
)

# 创建损失函数
weights = WeightedCrossEntropyLoss.calculate_class_weights(
    train_labels,
    num_classes=config['model']['num_classes']
)
criterion = WeightedCrossEntropyLoss(class_weights=weights)

# 创建优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(config['training']['num_epochs']):
    avg_loss, metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
```

---

## 注意事项

1. **输入维度**: 确保输入数据维度正确，卫星流为 (Batch, 1, Bands, 13, 13)，地面流为 (Batch, 1, Bands, 1, 1)
2. **设备管理**: 模型和数据需要在同一设备上（CPU 或 GPU）
3. **批量大小**: 建议批量大小在 1-128 之间，根据 GPU 内存调整
4. **数据归一化**: 建议对输入数据进行归一化处理
5. **类别权重**: 对于不平衡数据集，建议使用加权损失函数

---

## 版本信息

- **当前版本**: 0.1.0 (Alpha)
- **PyTorch 版本**: 2.0+
- **Python 版本**: 3.8+

---

## 参考文档

- [需求文档](.kiro/specs/ag-s2cnn/requirements.md)
- [设计文档](.kiro/specs/ag-s2cnn/design.md)
- [配置使用](config_cli_usage.md)
- [训练流水线](training_pipeline_usage.md)
