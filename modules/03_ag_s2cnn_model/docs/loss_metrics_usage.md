# 损失函数与评估指标使用指南

本文档介绍如何使用 AG-S²CNN 项目中的损失函数和评估指标模块。

## 目录

1. [加权交叉熵损失函数](#加权交叉熵损失函数)
2. [评估指标计算器](#评估指标计算器)
3. [与模型集成](#与模型集成)
4. [完整训练示例](#完整训练示例)

---

## 加权交叉熵损失函数

`WeightedCrossEntropyLoss` 用于处理类别不平衡问题，对稀有类别赋予更高的权重。

### 基本使用

```python
from utils.loss import WeightedCrossEntropyLoss
import torch

# 创建标准交叉熵损失（不使用权重）
criterion = WeightedCrossEntropyLoss()

# 计算损失
predictions = torch.randn(8, 4)  # (batch_size, num_classes)
targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
loss = criterion(predictions, targets)
```

### 计算类别权重

```python
import numpy as np

# 训练集标签
train_labels = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])

# 计算权重
weights = WeightedCrossEntropyLoss.calculate_class_weights(
    train_labels, 
    num_classes=4
)

# 创建加权损失函数
criterion = WeightedCrossEntropyLoss(class_weights=weights)
```

### 权重计算公式

权重计算使用以下公式：

```
w_c = N_total / (k × N_c)
```

其中：
- `N_total`: 总样本数
- `k`: 类别数
- `N_c`: 类别 c 的样本数

对于零样本类别，权重设为 0。

### 处理零样本类别

```python
# 类别 2 没有样本
labels = np.array([0, 0, 1, 1, 3, 3, 3])
weights = WeightedCrossEntropyLoss.calculate_class_weights(labels, num_classes=4)

# weights[2] 将为 0.0
print(f"类别 2 的权重: {weights[2]}")  # 输出: 0.0
```

---

## 评估指标计算器

`MetricsCalculator` 提供多维度的评估指标，包括：

- **Overall Accuracy (OA)**: 总体精度
- **Average Accuracy (AA)**: 平均精度
- **Kappa 系数**: Cohen's Kappa 系数
- **F1-Score**: 每个类别的 F1 分数
- **精确率和召回率**: 每个类别的详细指标
- **混淆矩阵**: 预测结果的混淆矩阵

### 基本使用

```python
from utils.metrics import MetricsCalculator
import numpy as np

# 创建计算器
calculator = MetricsCalculator(num_classes=4)

# 真实标签和预测结果
y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
y_pred = np.array([0, 1, 2, 3, 0, 2, 2, 3])

# 计算所有指标
metrics = calculator.calculate_all_metrics(y_true, y_pred)

# 访问指标
print(f"Overall Accuracy: {metrics['OA']:.4f}")
print(f"Kappa: {metrics['Kappa']:.4f}")
print(f"F1-Score (Macro): {metrics['F1_macro']:.4f}")
```

### 打印格式化报告

```python
# 定义类别名称
class_names = ['锂辉石伟晶岩', '贫矿伟晶岩', '围岩', '背景']

# 打印详细指标报告
calculator.print_metrics(metrics, class_names)

# 打印混淆矩阵
calculator.print_confusion_matrix(metrics, class_names)
```

### 指标字典结构

`calculate_all_metrics()` 返回的字典包含：

```python
{
    'OA': float,                      # Overall Accuracy
    'AA': float,                      # Average Accuracy
    'Kappa': float,                   # Cohen's Kappa
    'F1_per_class': ndarray,          # 每个类别的 F1-Score
    'F1_macro': float,                # 宏平均 F1-Score
    'Precision_per_class': ndarray,   # 每个类别的精确率
    'Recall_per_class': ndarray,      # 每个类别的召回率
    'confusion_matrix': ndarray       # 混淆矩阵 (num_classes, num_classes)
}
```

---

## 与模型集成

### 完整的训练和验证流程

```python
import torch
from models.ag_s2cnn import AG_S2CNN
from utils.loss import WeightedCrossEntropyLoss
from utils.metrics import MetricsCalculator

# 创建模型
model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=4)

# 创建损失函数（假设已计算权重）
criterion = WeightedCrossEntropyLoss(class_weights=weights)

# 创建评估指标计算器
calculator = MetricsCalculator(num_classes=4)

# 训练模式
model.train()
for batch in train_loader:
    x_sat, x_ref, targets = batch
    
    # 前向传播
    outputs = model(x_sat, x_ref)
    
    # 计算损失
    loss = criterion(outputs, targets)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 验证模式
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch in val_loader:
        x_sat, x_ref, targets = batch
        
        # 前向传播
        outputs = model(x_sat, x_ref)
        
        # 获取预测结果
        predictions = torch.argmax(outputs, dim=1)
        
        all_predictions.append(predictions.numpy())
        all_targets.append(targets.numpy())

# 计算评估指标
y_true = np.concatenate(all_targets)
y_pred = np.concatenate(all_predictions)
metrics = calculator.calculate_all_metrics(y_true, y_pred)

# 打印结果
calculator.print_metrics(metrics, class_names)
```

---

## 完整训练示例

以下是一个完整的训练循环示例：

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.ag_s2cnn import AG_S2CNN
from utils.dataset import HyperspectralDataset
from utils.loss import WeightedCrossEntropyLoss
from utils.metrics import MetricsCalculator

# 配置
num_epochs = 100
batch_size = 32
learning_rate = 0.001
num_classes = 4

# 创建数据集和数据加载器
train_dataset = HyperspectralDataset(
    image_cube, ground_truth, gsrsl,
    spatial_size=13, augmentation=True
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size,
    shuffle=True, num_workers=4
)

# 计算类别权重
train_labels = [label for _, _, label in train_dataset]
weights = WeightedCrossEntropyLoss.calculate_class_weights(
    train_labels, num_classes
)

# 创建模型、损失函数和优化器
model = AG_S2CNN(num_bands=290, spatial_size=13, num_classes=num_classes)
criterion = WeightedCrossEntropyLoss(class_weights=weights)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# 创建评估指标计算器
calculator = MetricsCalculator(num_classes=num_classes)

# 训练循环
best_oa = 0.0
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    
    for batch_idx, (x_sat, x_ref, targets) in enumerate(train_loader):
        # 前向传播
        outputs = model(x_sat, x_ref)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # 验证阶段
    model.eval()
    all_predictions = []
    all_targets = []
    val_loss = 0.0
    
    with torch.no_grad():
        for x_sat, x_ref, targets in val_loader:
            outputs = model(x_sat, x_ref)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.append(predictions.numpy())
            all_targets.append(targets.numpy())
    
    # 计算评估指标
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    metrics = calculator.calculate_all_metrics(y_true, y_pred)
    
    # 打印结果
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"  OA: {metrics['OA']:.4f}")
    print(f"  Kappa: {metrics['Kappa']:.4f}")
    
    # 保存最佳模型
    if metrics['OA'] > best_oa:
        best_oa = metrics['OA']
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  保存最佳模型 (OA: {best_oa:.4f})")
    
    # 更新学习率
    scheduler.step()

print("训练完成！")
```

---

## 注意事项

1. **类别权重计算**: 在训练开始前计算一次即可，不需要每个 epoch 重新计算
2. **零样本类别**: 如果某个类别在训练集中没有样本，其权重会被设为 0
3. **评估指标**: 建议在验证集和测试集上都计算完整的评估指标
4. **混淆矩阵**: 混淆矩阵可以帮助识别哪些类别容易被混淆
5. **早停机制**: 可以基于验证集的 OA 或 Kappa 系数实现早停

---

## 参考

- 需求文档: `.kiro/specs/ag-s2cnn/requirements.md`
- 设计文档: `.kiro/specs/ag-s2cnn/design.md`
- 示例代码: `examples/demo_loss_metrics.py`
- 测试代码: `tests/test_loss.py`, `tests/test_metrics.py`
