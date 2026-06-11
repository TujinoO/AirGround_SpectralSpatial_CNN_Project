# AG-S²CNN 文档索引

本文档提供 AG-S²CNN 项目所有文档的快速导航。

## 📚 核心文档

### 项目概述
- **[README.md](../README.md)** - 项目简介、安装说明和快速开始指南

### 规范文档
- **[需求文档](.kiro/specs/ag-s2cnn/requirements.md)** - 详细的功能需求和验收标准
- **[设计文档](.kiro/specs/ag-s2cnn/design.md)** - 系统架构、组件接口和正确性属性
- **[任务列表](.kiro/specs/ag-s2cnn/tasks.md)** - 实现计划和任务追踪

## 📖 API 文档

### 完整 API 参考
- **[API 文档](api_documentation.md)** - 所有公共类和方法的详细 API 参考
  - 模型模块 (AG_S2CNN, GEncoder, SBackbone, etc.)
  - 工具模块 (Dataset, Loss, Metrics, Config)
  - 训练模块 (train_one_epoch, validate, EarlyStopping)

## 🛠️ 使用指南

### 配置和命令行
- **[配置和命令行使用](config_cli_usage.md)** - 配置文件格式和命令行参数说明

### 训练流水线
- **[训练流水线使用](training_pipeline_usage.md)** - 完整的训练流程和最佳实践

### 损失和指标
- **[损失和指标使用](loss_metrics_usage.md)** - 损失函数和评估指标的使用方法

### 日志和可视化
- **[日志和可视化使用](logger_visualization_usage.md)** - 日志系统和可视化工具的使用

### 错误处理
- **[错误处理实现](error_handling_implementation.md)** - 错误处理策略和故障排除

## 💡 示例代码

### 示例目录
- **[示例 README](../examples/README.md)** - 所有示例的概述和使用说明

### 具体示例
1. **[数据预处理示例](../examples/demo_data_preprocessing.py)**
   - 加载和预处理高光谱影像
   - 创建 Ground Truth 和 GSRSL
   - 数据归一化和可视化

2. **[训练示例](../examples/demo_training.py)**
   - 完整的训练流程
   - 早停机制和模型保存
   - 训练曲线和混淆矩阵生成

3. **[推理示例](../examples/demo_inference.py)**
   - 单样本推理
   - 全图推理
   - 结果可视化

4. **[损失和指标示例](../examples/demo_loss_metrics.py)**
   - 加权损失函数使用
   - 评估指标计算

5. **[日志和可视化示例](../examples/demo_logger_visualization.py)**
   - 日志配置
   - 可视化工具使用


## 📋 快速导航

### 新手入门
1. 阅读 [README.md](../README.md) 了解项目概述
2. 查看 [示例 README](../examples/README.md) 了解使用流程
3. 运行 [数据预处理示例](../examples/demo_data_preprocessing.py)
4. 运行 [训练示例](../examples/demo_training.py)

### 深入学习
1. 阅读 [需求文档](.kiro/specs/ag-s2cnn/requirements.md) 了解系统需求
2. 阅读 [设计文档](.kiro/specs/ag-s2cnn/design.md) 了解架构设计
3. 查看 [API 文档](api_documentation.md) 了解详细接口

### 问题排查
1. 查看 [错误处理文档](error_handling_implementation.md)
2. 检查 [配置使用文档](config_cli_usage.md)
3. 参考 [训练流水线文档](training_pipeline_usage.md)

## 📊 文档结构

```
ag-s2cnn/
├── README.md                          # 项目主文档
├── .kiro/specs/ag-s2cnn/
│   ├── requirements.md                # 需求文档
│   ├── design.md                      # 设计文档
│   └── tasks.md                       # 任务列表
├── docs/
│   ├── documentation_index.md         # 本文档（文档索引）
│   ├── api_documentation.md           # API 参考文档
│   ├── config_cli_usage.md            # 配置和命令行使用
│   ├── training_pipeline_usage.md     # 训练流水线使用
│   ├── loss_metrics_usage.md          # 损失和指标使用
│   ├── logger_visualization_usage.md  # 日志和可视化使用
│   ├── error_handling_implementation.md # 错误处理实现
│   └── task_14_implementation_summary.md # 任务实现总结
└── examples/
    ├── README.md                      # 示例概述
    ├── demo_data_preprocessing.py     # 数据预处理示例
    ├── demo_training.py               # 训练示例
    ├── demo_inference.py              # 推理示例
    ├── demo_loss_metrics.py           # 损失和指标示例
    └── demo_logger_visualization.py   # 日志和可视化示例
```

## 🔍 按主题查找

### 数据处理
- [数据预处理示例](../examples/demo_data_preprocessing.py)
- [数据集 API](api_documentation.md#hyperspectraldataset)
- [数据增强 API](api_documentation.md#dataaugmentation)

### 模型训练
- [训练示例](../examples/demo_training.py)
- [训练流水线文档](training_pipeline_usage.md)
- [配置使用文档](config_cli_usage.md)

### 模型推理
- [推理示例](../examples/demo_inference.py)
- [模型 API](api_documentation.md#ag_s2cnn)

### 评估和可视化
- [损失和指标示例](../examples/demo_loss_metrics.py)
- [日志和可视化示例](../examples/demo_logger_visualization.py)
- [指标 API](api_documentation.md#metricscalculator)

### 配置管理
- [配置使用文档](config_cli_usage.md)
- [配置 API](api_documentation.md#config)

### 错误处理
- [错误处理文档](error_handling_implementation.md)
- [故障排除指南](../README.md#故障排除)

## 📝 文档贡献

如果您发现文档中的错误或希望改进文档，请：
1. 提交 Issue 描述问题
2. 提交 Pull Request 修复问题
3. 联系项目维护者

## 📌 版本信息

- **文档版本**: 1.0.0
- **项目版本**: 0.1.0 (Alpha)
- **最后更新**: 2024

## 🔗 相关链接

- [PyTorch 文档](https://pytorch.org/docs/)
- [NumPy 文档](https://numpy.org/doc/)
- [Scikit-learn 文档](https://scikit-learn.org/stable/)

---

**提示**: 使用 Ctrl+F (或 Cmd+F) 在本页面快速搜索您需要的文档。
