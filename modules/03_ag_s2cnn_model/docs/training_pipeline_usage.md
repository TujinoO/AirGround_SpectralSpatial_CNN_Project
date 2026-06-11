# 训练流水线使用指南

本文档介绍如何使用 AG-S²CNN 模型的训练流水线。

## 概述

训练流水线包含以下核心功能：

1. **训练主循环** - 训练一个 epoch 的完整流程
2. **验证函数** - 在验证集上评估模型性能
3. **早停机制** - 防止过拟合，自动停止训练
4. **模型保存与加载** - 保存和恢复训练状态
5. **设备管理** - 自动检测和使用 GPU/CPU

## 快速开始

### 1. 准备配置文件

创建 `config.yaml` 文件：

```yaml
# 模型配置
model:
  num_bands: 290
  spatial_size: 13
  num_classes: 4

# 训练配置
training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  num_epochs: 100
  early_stopping_patience: 20

# 数据配置
data:
  image_path: "data/gf5_image.npy"
  ground_truth_path: "data/ground_truth.npy"
  gsrsl_path: "data/gsrsl.pkl"
  output_dir: "./outputs"

# 设备配置
device:
  use_cuda: true
  gpu_id: 0
```

### 2. 准备数据

```python
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import HyperspectralDataset

# 加载数据
image_cube = np.load('data/gf5_image.npy')  # (H, W, Bands)
ground_truth = np.load('data/ground_truth.npy')  # (H, W)
gsrsl = np.load('data/gsrsl.pkl', allow_pickle=True)  # {class_id: spectrum}

# 创建数据集
train_dataset = HyperspectralDataset(
    image_cube=image_cube,
    ground_truth=ground_truth,
    gsrsl=gsrsl,
    spatial_size=13,
    augmentation=True  # 训练时启用数据增强
)

val_dataset = HyperspectralDataset(
    image_cube=image_cube,
    ground_truth=ground_truth,
    gsrsl=gsrsl,
    spatial_size=13,
    augmentation=False  # 验证时不使用数据增强
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)
```

### 3. 运行训练

#### 方式 1: 使用命令行脚本

```bash
python train.py
```

#### 方式 2: 自定义训练循环

```python
import torch
from models.ag_s2cnn import AG_S2CNN
from utils.loss import WeightedCrossEntropyLoss
from utils.metrics import MetricsCalculator
from train import (
    train_one_epoch,
    validate,
    EarlyStopping,
    save_checkpoint,
    get_device
)
from config import Config

# 加载配置
config = Config.from_yaml('config.yaml')

# 获取设备
device = get_device(use_cuda=True, gpu_id=0)

# 创建模型
model = AG_S2CNN(
    num_bands=290,
    spatial_size=13,
    num_classes=4
).to(device)

# 创建优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

# 创建学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)

# 创建损失函数（可选：计算类别权重）
# labels = [sample[2] for sample in train_dataset.samples]
# class_weights = WeightedCrossEntropyLoss.calculate_class_weights(labels, num_classes=4)
# criterion = WeightedCrossEntropyLoss(class_weights=class_weights.to(device))
criterion = WeightedCrossEntropyLoss()

# 创建评估指标计算器
metrics_calculator = MetricsCalculator(num_classes=4)

# 创建早停机制
early_stopping = EarlyStopping(patience=20, mode='max')

# 训练循环
best_val_acc = 0.0
for epoch in range(1, 101):
    print(f"\nEpoch {epoch}/100")
    
    # 训练
    train_loss, train_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    
    # 验证
    val_loss, val_metrics = validate(
        model, val_loader, criterion, device, metrics_calculator, epoch
    )
    
    # 更新学习率
    scheduler.step()
    
    # 打印指标
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val OA: {val_metrics['OA']:.4f}, "
          f"Val Kappa: {val_metrics['Kappa']:.4f}")
    
    # 保存最佳模型
    val_acc = val_metrics['OA']
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics, config,
            filepath='outputs/checkpoints/best_model.pth'
        )
        print(f"最佳模型已保存 (Val OA: {val_acc:.4f})")
    
    # 早停检查
    if early_stopping(val_acc):
        print(f"\n早停触发: 验证集精度连续 {early_stopping.patience} 个 epoch 未提升")
        break

print(f"\n训练完成！最佳验证精度: {best_val_acc:.4f}")
```

## 核心功能详解

### 训练一个 Epoch

```python
from train import train_one_epoch

avg_loss, metrics = train_one_epoch(
    model=model,
    dataloader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epoch=1
)

print(f"平均损失: {avg_loss:.4f}")
print(f"训练精度: {metrics['accuracy']:.4f}")
```

### 验证模型

```python
from train import validate

avg_loss, metrics = validate(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    device=device,
    metrics_calculator=metrics_calculator,
    epoch=1
)

print(f"验证损失: {avg_loss:.4f}")
print(f"Overall Accuracy: {metrics['OA']:.4f}")
print(f"Average Accuracy: {metrics['AA']:.4f}")
print(f"Kappa: {metrics['Kappa']:.4f}")
print(f"F1-Score (Macro): {metrics['F1_macro']:.4f}")
```

### 早停机制

```python
from train import EarlyStopping

# 创建早停对象（监控验证集精度，越大越好）
early_stopping = EarlyStopping(patience=20, mode='max')

for epoch in range(num_epochs):
    # ... 训练和验证 ...
    
    # 检查是否应该早停
    if early_stopping(val_metrics['OA']):
        print(f"早停触发！连续 {early_stopping.patience} 个 epoch 未提升")
        break
```

也可以监控损失（越小越好）：

```python
early_stopping = EarlyStopping(patience=20, mode='min')

for epoch in range(num_epochs):
    # ... 训练和验证 ...
    
    if early_stopping(val_loss):
        print("早停触发！")
        break
```

### 模型保存与加载

#### 保存模型

```python
from train import save_checkpoint

save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=50,
    metrics={'OA': 0.85, 'Kappa': 0.80},
    config=config,
    filepath='outputs/checkpoints/model_epoch_50.pth'
)
```

#### 加载模型

```python
from train import load_checkpoint

checkpoint = load_checkpoint(
    filepath='outputs/checkpoints/best_model.pth',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device
)

print(f"已加载 Epoch {checkpoint['epoch']}")
print(f"指标: {checkpoint['metrics']}")

# 继续训练
start_epoch = checkpoint['epoch'] + 1
```

### 设备管理

```python
from train import get_device

# 自动检测并使用 GPU
device = get_device(use_cuda=True, gpu_id=0)

# 强制使用 CPU
device = get_device(use_cuda=False)

# 使用多个 GPU（需要额外配置）
device = get_device(use_cuda=True, gpu_id=1)
```

## 优化器和学习率调度

### AdamW 优化器

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,           # 学习率
    weight_decay=1e-4  # 权重衰减（L2 正则化）
)
```

### 余弦退火学习率调度

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,    # 最大 epoch 数
    eta_min=1e-6  # 最小学习率
)

# 在每个 epoch 结束后调用
scheduler.step()

# 查看当前学习率
current_lr = scheduler.get_last_lr()[0]
print(f"当前学习率: {current_lr:.6f}")
```

## 加权损失函数

处理类别不平衡问题：

```python
from utils.loss import WeightedCrossEntropyLoss
import numpy as np

# 收集所有训练标签
train_labels = [sample[2] for sample in train_dataset.samples]

# 计算类别权重
class_weights = WeightedCrossEntropyLoss.calculate_class_weights(
    labels=train_labels,
    num_classes=4
)

print(f"类别权重: {class_weights}")

# 创建加权损失函数
criterion = WeightedCrossEntropyLoss(class_weights=class_weights.to(device))
```

## 评估指标

```python
from utils.metrics import MetricsCalculator

metrics_calculator = MetricsCalculator(num_classes=4)

# 计算所有指标
metrics = metrics_calculator.calculate_all_metrics(y_true, y_pred)

# 打印格式化报告
class_names = ['锂辉石伟晶岩', '贫矿伟晶岩', '围岩', '背景']
metrics_calculator.print_metrics(metrics, class_names)

# 打印混淆矩阵
metrics_calculator.print_confusion_matrix(metrics, class_names)
```

## 常见问题

### 1. GPU 内存不足

如果遇到 CUDA OOM 错误，可以：

- 减小批量大小
- 使用梯度累积
- 使用混合精度训练

```python
# 减小批量大小
train_loader = DataLoader(train_dataset, batch_size=16, ...)

# 梯度累积示例
accumulation_steps = 4
for i, (x_sat, x_ref, targets) in enumerate(train_loader):
    outputs = model(x_sat, x_ref)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. 训练速度慢

优化建议：

- 使用多进程数据加载：`num_workers=4`
- 启用 pin_memory：`pin_memory=True`
- 使用 GPU 训练
- 减小模型复杂度

### 3. 过拟合

防止过拟合的方法：

- 启用数据增强
- 使用早停机制
- 增加权重衰减
- 使用 Dropout（如果模型支持）

### 4. 欠拟合

提升模型性能：

- 增加训练轮数
- 调整学习率
- 检查数据质量
- 调整模型架构

## 参考

- 需求文档: `.kiro/specs/ag-s2cnn/requirements.md`
- 设计文档: `.kiro/specs/ag-s2cnn/design.md`
- 任务列表: `.kiro/specs/ag-s2cnn/tasks.md`
