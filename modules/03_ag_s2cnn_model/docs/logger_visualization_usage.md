# 日志和可视化功能使用指南

本文档介绍如何使用 AG-S²CNN 项目中的日志记录系统和可视化功能。

## 目录

1. [日志记录系统](#日志记录系统)
2. [可视化功能](#可视化功能)
3. [集成使用示例](#集成使用示例)
4. [最佳实践](#最佳实践)

---

## 日志记录系统

日志记录系统提供了完整的训练过程记录功能，支持同时输出到控制台和文件。

### 基本使用

```python
from utils.logger import create_logger

# 创建日志记录器
logger = create_logger(
    log_dir='logs',              # 日志文件保存目录
    log_name='training',         # 日志文件名（不含扩展名）
    console_level=logging.INFO,  # 控制台日志级别
    file_level=logging.DEBUG     # 文件日志级别
)

# 记录基本信息
logger.info("这是一条信息")
logger.warning("这是一条警告")
logger.error("这是一条错误")

# 关闭日志记录器
logger.close()
```

### 记录配置信息

```python
config = {
    'num_bands': 290,
    'spatial_size': 13,
    'num_classes': 4,
    'batch_size': 32,
    'learning_rate': 0.001
}
logger.log_config(config)
```

### 记录设备信息

```python
logger.log_device_info('cuda:0', 'NVIDIA RTX 5060')
```

### 记录模型信息

```python
logger.log_model_info(
    num_params=790000,      # 模型参数量
    model_size_mb=3.2       # 模型大小（MB）
)
```

### 记录训练过程

```python
# 记录 epoch 开始
logger.log_epoch_start(epoch=1, total_epochs=100)

# 记录训练指标
train_metrics = {
    'loss': 0.5,
    'accuracy': 0.85
}
logger.log_training_metrics(epoch=1, metrics=train_metrics)

# 记录验证指标
val_metrics = {
    'loss': 0.6,
    'OA': 0.82,
    'AA': 0.78,
    'Kappa': 0.75,
    'F1_macro': 0.80
}
logger.log_validation_metrics(epoch=1, metrics=val_metrics)

# 记录学习率
logger.log_learning_rate(epoch=1, lr=0.001)
```

### 记录模型保存

```python
logger.log_model_save(
    filepath='checkpoints/best_model.pth',
    metrics={'OA': 0.85, 'Kappa': 0.80}
)
```

### 记录早停

```python
logger.log_early_stopping(
    epoch=50,
    patience=20,
    best_score=0.85
)
```

### 记录训练完成

```python
best_metrics = {
    'OA': 0.85,
    'AA': 0.82,
    'Kappa': 0.80,
    'F1_macro': 0.83
}
logger.log_training_complete(
    total_epochs=100,
    best_metrics=best_metrics
)
```

### 记录异常

```python
try:
    # 训练代码
    pass
except Exception as e:
    logger.log_exception(e)
    raise
```

---

## 可视化功能

可视化功能提供了多种图表生成功能，用于分析训练过程和模型性能。

### 基本使用

```python
from utils.visualization import create_visualizer

# 创建可视化器
visualizer = create_visualizer(
    save_dir='results',                    # 图表保存目录
    dpi=300,                               # 图像分辨率
    style='seaborn-v0_8-darkgrid'         # Matplotlib 样式
)

# 关闭所有图形
visualizer.close_all()
```

### 绘制损失曲线

```python
train_losses = [0.8, 0.7, 0.6, 0.5, 0.4]
val_losses = [0.85, 0.75, 0.65, 0.55, 0.45]

visualizer.plot_loss_curves(
    train_losses=train_losses,
    val_losses=val_losses,
    save_name='loss_curves.png',
    title='训练和验证损失曲线'
)
```

**输出**: `results/loss_curves.png`

### 绘制学习率曲线

```python
learning_rates = [0.001, 0.0008, 0.0006, 0.0004, 0.0002]

visualizer.plot_learning_rate_curve(
    learning_rates=learning_rates,
    save_name='learning_rate_curve.png',
    title='学习率变化曲线'
)
```

**输出**: `results/learning_rate_curve.png`

### 绘制混淆矩阵

```python
import numpy as np

confusion_matrix = np.array([
    [85, 8, 5, 2],
    [6, 78, 10, 6],
    [4, 7, 82, 7],
    [3, 5, 8, 84]
])

class_names = ['锂辉石伟晶岩', '贫矿伟晶岩', '围岩', '背景']

# 绘制原始混淆矩阵
visualizer.plot_confusion_matrix(
    confusion_matrix=confusion_matrix,
    class_names=class_names,
    save_name='confusion_matrix.png',
    title='混淆矩阵'
)

# 绘制归一化混淆矩阵（百分比）
visualizer.plot_confusion_matrix(
    confusion_matrix=confusion_matrix,
    class_names=class_names,
    save_name='confusion_matrix_normalized.png',
    title='混淆矩阵（归一化）',
    normalize=True
)
```

**输出**: 
- `results/confusion_matrix.png`
- `results/confusion_matrix_normalized.png`

### 绘制指标对比图

```python
metrics_dict = {
    'OA': [0.70, 0.75, 0.78, 0.80, 0.82],
    'AA': [0.68, 0.72, 0.75, 0.77, 0.79],
    'Kappa': [0.60, 0.65, 0.68, 0.70, 0.72],
    'F1_macro': [0.69, 0.73, 0.76, 0.78, 0.80]
}

visualizer.plot_metrics_comparison(
    metrics_dict=metrics_dict,
    save_name='metrics_comparison.png',
    title='评估指标对比'
)
```

**输出**: `results/metrics_comparison.png`

### 绘制类别性能图

```python
precision = np.array([0.85, 0.78, 0.82, 0.88])
recall = np.array([0.83, 0.75, 0.80, 0.86])
f1_scores = np.array([0.84, 0.76, 0.81, 0.87])
class_names = ['锂辉石伟晶岩', '贫矿伟晶岩', '围岩', '背景']

visualizer.plot_class_performance(
    precision=precision,
    recall=recall,
    f1_scores=f1_scores,
    class_names=class_names,
    save_name='class_performance.png',
    title='各类别性能指标'
)
```

**输出**: `results/class_performance.png`

---

## 集成使用示例

在训练流水线中集成日志和可视化功能：

```python
from utils.logger import create_logger
from utils.visualization import create_visualizer

# 创建日志记录器和可视化器
logger = create_logger(log_dir='logs', log_name='training')
visualizer = create_visualizer(save_dir='results')

# 记录配置
logger.log_config(config)
logger.log_device_info(device, device_name)
logger.log_model_info(num_params, model_size_mb)

# 训练循环
train_losses = []
val_losses = []
learning_rates = []

for epoch in range(1, num_epochs + 1):
    # 记录 epoch 开始
    logger.log_epoch_start(epoch, num_epochs)
    
    # 训练
    train_loss, train_metrics = train_one_epoch(...)
    train_losses.append(train_loss)
    logger.log_training_metrics(epoch, train_metrics)
    
    # 验证
    val_loss, val_metrics = validate(...)
    val_losses.append(val_loss)
    logger.log_validation_metrics(epoch, val_metrics)
    
    # 记录学习率
    lr = scheduler.get_last_lr()[0]
    learning_rates.append(lr)
    logger.log_learning_rate(epoch, lr)
    
    # 保存最佳模型
    if val_metrics['OA'] > best_oa:
        best_oa = val_metrics['OA']
        save_checkpoint(...)
        logger.log_model_save(checkpoint_path, val_metrics)
    
    # 早停检查
    if early_stopping(val_metrics['OA']):
        logger.log_early_stopping(epoch, patience, best_oa)
        break

# 训练完成后生成可视化
logger.info("生成可视化图表...")

visualizer.plot_loss_curves(train_losses, val_losses)
visualizer.plot_learning_rate_curve(learning_rates)
visualizer.plot_confusion_matrix(confusion_matrix, class_names)

# 记录训练完成
logger.log_training_complete(epoch, best_metrics)

# 清理
logger.close()
visualizer.close_all()
```

---

## 最佳实践

### 1. 日志级别设置

- **控制台**: 使用 `INFO` 级别，显示关键信息
- **文件**: 使用 `DEBUG` 级别，记录详细信息

```python
logger = create_logger(
    console_level=logging.INFO,
    file_level=logging.DEBUG
)
```

### 2. 日志文件命名

使用时间戳或实验名称命名日志文件：

```python
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = create_logger(log_name=f'training_{timestamp}')
```

### 3. 异常处理

在训练循环中使用 try-except 捕获异常：

```python
try:
    for epoch in range(num_epochs):
        train_one_epoch(...)
        validate(...)
except KeyboardInterrupt:
    logger.warning("训练被用户中断")
    save_checkpoint(...)
except Exception as e:
    logger.log_exception(e)
    raise
finally:
    logger.close()
    visualizer.close_all()
```

### 4. 定期保存可视化

在训练过程中定期生成可视化图表：

```python
if epoch % 10 == 0:
    visualizer.plot_loss_curves(train_losses, val_losses)
    visualizer.plot_learning_rate_curve(learning_rates)
```

### 5. 内存管理

在生成大量图表后关闭图形以释放内存：

```python
visualizer.plot_loss_curves(...)
visualizer.plot_learning_rate_curve(...)
visualizer.close_all()  # 关闭所有图形
```

### 6. 中文字体支持

如果遇到中文显示问题，可以在代码中设置字体：

```python
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

### 7. 图表分辨率

根据用途设置合适的 DPI：

- **屏幕显示**: 72-96 DPI
- **打印**: 300 DPI
- **出版**: 600 DPI

```python
visualizer = create_visualizer(dpi=300)  # 适合打印
```

---

## 示例输出

### 日志文件示例

```
2026-01-25 11:25:47 - INFO - ============================================================
2026-01-25 11:25:47 - INFO - 日志系统初始化完成
2026-01-25 11:25:47 - INFO - 日志文件: logs/training.log
2026-01-25 11:25:47 - INFO - ============================================================
2026-01-25 11:25:47 - INFO - 
2026-01-25 11:25:47 - INFO - ============================================================
2026-01-25 11:25:47 - INFO - Epoch 1/100
2026-01-25 11:25:47 - INFO - ============================================================
2026-01-25 11:25:47 - INFO - [Epoch 1] 训练指标:
2026-01-25 11:25:47 - INFO -   loss: 0.700000
2026-01-25 11:25:47 - INFO -   accuracy: 0.650000
2026-01-25 11:25:47 - INFO - [Epoch 1] 验证指标:
2026-01-25 11:25:47 - INFO -   loss: 0.750000
2026-01-25 11:25:47 - INFO -   OA: 0.630000
2026-01-25 11:25:47 - INFO -   Kappa: 0.550000
```

### 可视化图表示例

生成的图表包括：

1. **损失曲线图**: 显示训练损失和验证损失随 epoch 的变化
2. **学习率曲线图**: 显示学习率调度策略的效果
3. **混淆矩阵**: 显示模型在各类别上的预测性能
4. **指标对比图**: 显示多个评估指标的变化趋势
5. **类别性能图**: 显示各类别的精确率、召回率和 F1 分数

---

## 相关文档

- [训练流水线使用指南](training_pipeline_usage.md)
- [损失函数和评估指标使用指南](loss_metrics_usage.md)
- [配置管理指南](../config.yaml)

---

## 需求追溯

本模块实现了以下需求：

- **需求 14.1**: 配置 Python logging 模块
- **需求 14.2**: 同时输出到控制台和文件
- **需求 14.3**: 生成损失曲线图（Matplotlib）
- **需求 14.4**: 生成学习率变化曲线图
- **需求 14.5**: 生成混淆矩阵热力图（Seaborn）
- **需求 14.7**: 记录训练损失、验证损失和评估指标
