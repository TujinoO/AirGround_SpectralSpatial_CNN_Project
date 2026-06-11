# AG-S2CNN: Air-Ground Spectral-Spatial CNN

AG-S2CNN（空地协同光谱-空间卷积神经网络）面向高海拔荒漠区伟晶岩型锂矿预测，融合卫星高光谱影像与地面标准光谱，在二分类目标检出场景下完成训练、推理、可视化与模型评价。

## 仓库概览

- 主任务：以 `Rich Ore Pegmatite` 为目标类，完成目标/背景二分类。
- 主流程：`train.py` 负责训练与验证，`predict.py` 负责全图推理与制图。
- 模型评价：`evaluation/` 独立承载对比实验、消融实验及其报告绘图。
- 文档与示例：`docs/` 提供专题说明，`examples/` 提供最小调用示例。

## 核心特性

- 空地协同双流架构：同时编码卫星影像与地面参考光谱。
- 物理约束融合：通过差分融合抑制背景干扰，提升目标可分性。
- 强化损失设计：支持加权 BCE、困难负样本挖掘与 separation margin。
- 稠密推理与论文制图：支持概率图、叠加图、激活图与 GeoTIFF 地理坐标绘图。
- 独立评价模块：支持模型对比、结构消融、损失消融与参数敏感性分析。

## 目录结构

```text
Air-Ground_Spectral-Spatial_CNN/
├── evaluation/                  # 模型评价模块
│   ├── core.py                  # 评价公共能力：数据、指标、绘图、统计
│   ├── models.py                # 对比/消融所需模型构建器
│   ├── run_comparative_experiments.py
│   ├── run_ablation_experiments.py
│   ├── run_core_ablation_suite.py
│   └── README.md
├── models/                      # AG-S2CNN 主模型与网络层
├── utils/                       # 数据、损失、指标、日志、可视化工具
├── docs/                        # 主题文档
├── examples/                    # 示例脚本
├── tests/                       # 测试与验证脚本
├── train.py                     # 训练/验证/测试入口
├── predict.py                   # 独立推理入口
├── config.py                    # 配置管理
├── config.yaml                  # 主配置示例
├── config_ablation_quick.yaml   # 消融快速配置
├── run_comparative_experiments.py  # 兼容入口，转发到 evaluation/
├── run_ablation_experiments.py     # 兼容入口，转发到 evaluation/
├── run_core_ablation_suite.py      # 兼容入口，转发到 evaluation/
├── beautify_ablation_focus_plots.py
├── beautify_final_comparative_plots.py
└── README.md
```

## 安装

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- 建议使用支持 CUDA 的 GPU

### 安装依赖

```bash
pip install -r requirements.txt
```

或以可编辑模式安装：

```bash
pip install -e .
```

主要依赖包括 `torch`、`numpy`、`pandas`、`scikit-learn`、`matplotlib`、`seaborn`、`rasterio`、`pyyaml`、`tqdm`、`pytest`。

## 数据准备

需要准备以下输入：

- 高光谱影像：推荐 GeoTIFF，也支持 `.npy`
- Ground Truth 标签图：与影像空间尺寸一致
- 地面标准参考光谱库：`gsrsl.npy`，格式为 `{class_id: spectrum}`

典型配置位于 `config.yaml`：

```yaml
model:
  spatial_size: 25

training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 20
  hnm_ratio: 0.5
  hnm_weight: 0.2
  separation_weight: 0.2
  separation_margin: 0.06

labels:
  target_class_name: "Rich Ore Pegmatite"
  gt_target_labels: [1]
  gsrsl_target_labels: [0, 1, 2]
  background_class_index: 0

data:
  image_path: "D:/.../image.tif"
  ground_truth_path: "D:/.../label.tif"
  gsrsl_path: "D:/.../gsrsl.npy"
  output_dir: "D:/.../output"
```

## 快速开始

### 1. 训练

```bash
python train.py --config config.yaml --mode train
```

常见变体：

```bash
python train.py --config config.yaml --batch-size 16 --epochs 30 --lr 5e-4
python train.py --config config.yaml --resume D:/.../output/checkpoints/best_model.pth
```

### 2. 验证与测试

```bash
python train.py --config config.yaml --mode validate --checkpoint D:/.../output/checkpoints/best_model.pth
python train.py --config config.yaml --mode test --checkpoint D:/.../output/checkpoints/best_model.pth
```

### 3. 全图推理

```bash
python predict.py --config config.yaml --checkpoint D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/output/checkpoints/best_model.pth --predict-image D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/GF5A_AHSI_E89.3_N38.7_20240126_experiment.tif --predict-output-dir D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/output_Presentation
```

`predict.py` 会输出预测图、概率热力图、叠加图、激活图，并在输入具备地理参考时同步导出 GeoTIFF 结果。

## 模型评价模块

评价相关脚本已按模块化设计收拢到 `evaluation/` 下，推荐使用模块入口：

```bash
python -m evaluation.run_comparative_experiments --config config.yaml --suite deep --scheme AG-S2CNN --seeds 2025
python -m evaluation.run_ablation_experiments --config config.yaml --ablation-type Architecture --scheme Full --seeds 2025
python -m evaluation.run_core_ablation_suite --config config_ablation_quick.yaml --seeds 2025
```

兼容说明：

- 根目录同名脚本仍可继续使用。
- 新的推荐组织以 `evaluation/` 为准，便于后续继续扩展评价逻辑与绘图产物。

详细说明见 `evaluation/README.md`。

## 训练与推理输出

典型输出包括：

- `loss_curves.png`：训练/验证损失曲线
- `learning_rate_curve.png`：学习率曲线
- `confusion_matrix_test.png`、`confusion_matrix_test_normalized.png`
- `class_performance_test.png`
- `prediction_map_dense.png`
- `rich_ore_probability_map.png`
- `feature_activation_map.png`
- `ore_potential_overlay.png`
- `metrics_report.json`
- `config_used.yaml`

## 代码模块说明

- `models/`：AG-S2CNN 主模型、骨干网络、分类头及基础层定义
- `utils/`：数据加载、损失函数、指标计算、日志与可视化
- `evaluation/`：独立评价工程，负责对比实验、消融实验、统计检验与报告绘图
- `examples/`：训练、推理、日志、损失与数据预处理示例
- `docs/`：配置、训练、损失与错误处理文档
- `tests/`：单元测试、集成测试与检查点验证

## 测试

运行测试：

```bash
pytest tests/ -v
```

查看覆盖率：

```bash
pytest tests/ --cov=models --cov=utils --cov-report=html
```

## 文档索引

- `docs/config_cli_usage.md`
- `docs/training_pipeline_usage.md`
- `docs/loss_metrics_usage.md`
- `docs/error_handling_implementation.md`
- `evaluation/README.md`
- `examples/README.md`

## 开发说明

- 当前仓库包含部分结果样图与验证报告，便于快速回看实验产物。
- `beautify_ablation_focus_plots.py` 与 `beautify_final_comparative_plots.py` 用于结果后处理与论文级美化。
- 如需新增评价方案，优先扩展 `evaluation/core.py`、`evaluation/models.py` 与 `evaluation/` 下对应入口脚本。

## 故障排查

### CUDA 显存不足

- 减小 `batch_size`
- 减小 `spatial_size`
- 缩短训练轮数或降低增强强度

### 维度不匹配

- 确认卫星流输入为 `N x 1 x Bands x S x S`
- 确认地面流输入为 `N x 1 x Bands x 1 x 1`
- 确认配置中的 `spatial_size` 为正奇数

### 训练不收敛

- 检查数据归一化与标签映射
- 调整学习率与权重衰减
- 检查目标类样本量与背景采样比例

## 备注

- 许可证、引用与联系方式信息后续可按论文或项目发布要求补充。
