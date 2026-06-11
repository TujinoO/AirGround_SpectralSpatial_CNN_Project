# 模型评价模块说明

`evaluation/` 是 AG-S2CNN 的独立模型评价子模块，用于在训练与推理之外，统一完成对比实验、消融实验、统计检验与报告绘图，不侵入 `train.py` 和 `predict.py` 的主流程。

## 设计目标

- 对比实验：证明 AG-S2CNN 相对物理方法、浅层机器学习和常规深度学习基线的性能优势。
- 消融实验：解释性能提升来自哪些结构、损失和参数设计。
- 可复现：支持多随机种子、增量执行、自动汇总和持续更新图表。
- 可解释：不仅输出指标，还输出 ROC/PR 曲线、概率分布、排序表和散点权衡图。

## 模块结构

```text
evaluation/
├── core.py                         # 数据准备、指标、绘图、显著性检验
├── models.py                       # 对比/消融实验模型构建器
├── run_comparative_experiments.py  # 对比实验主入口
├── run_ablation_experiments.py     # 消融实验主入口
├── run_core_ablation_suite.py      # 核心四变体批量执行入口
└── README.md
```

分层职责：

- `core.py`
  - 统一加载数据与划分 train/val/test
  - 计算二分类指标与统计量
  - 生成柱状图、ROC/PR 曲线、概率直方图
  - 输出 JSON 索引与显著性检验结果
- `models.py`
  - 提供 AG-S2CNN、3D-CNN 基线和可配置消融网络
  - 统一对比实验与消融实验的模型构建接口
- `run_*.py`
  - 负责实验入口参数、单方案执行、结果累积与报告刷新

## 已整理的入口位置

原先位于仓库根目录的实验驱动脚本，现已按设计思路放置到本模块内：

- `evaluation/run_comparative_experiments.py`
- `evaluation/run_ablation_experiments.py`
- `evaluation/run_core_ablation_suite.py`

兼容说明：

- 仓库根目录仍保留同名轻量脚本作为兼容入口。
- 后续推荐统一从 `evaluation/` 模块入口启动。

## 实验方案

### 1. 对比实验

评估对象：

- 物理方法：`SAM` / `SID` / `SAM-SID` / `CEM`
- 机器学习：`LogisticRegression` / `SVM-RBF` / `RandomForest` / `MLP` / `XGBoost`（可选）
- 深度学习：`AG-S2CNN` / `3D-CNN-Standard`

统一流程：

1. 读取配置与数据
2. 按相同 seed 划分 train/val/test
3. 训练或构建单个方案
4. 计算测试集概率与指标
5. 追加写入明细结果
6. 立即刷新汇总表、排行榜、曲线图与显著性结果

### 2. 消融实验

支持以下消融方向：

- 结构消融：`Full`、`SmallOnly`、`LargeOnly`、`NoPhysics`、`PhysicsConcat`、`PhysicsDiffNoAtt`、`FCHead`、`LightHead`
- 损失消融：`StandardBCE`、`WeightedBCE`、`WeightedBCE_HNM`、`WeightedBCE_Margin`、`FullLoss`
- 参数敏感性：窗口大小、HNM ratio、margin

### 3. 核心四变体套件

`run_core_ablation_suite.py` 会顺序执行四个核心结构变体：

- `NoPhysics`
- `PhysicsConcat`
- `PhysicsDiffNoAtt`
- `Full`

适合快速获得论文主对比图所需的结构消融结果。

## 运行方式

推荐使用模块方式运行：

```bash
# 对比实验：每次运行一个方案，结果增量写入
python -m evaluation.run_comparative_experiments --config config.yaml --output-dir D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/evaluation_outputs/comparative --suite deep --scheme AG-S2CNN --seeds 2025
python -m evaluation.run_comparative_experiments --config config.yaml --output-dir D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/evaluation_outputs/comparative --suite deep --scheme 3D-CNN-Standard --seeds 2025
python -m evaluation.run_comparative_experiments --config config.yaml --output-dir D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/evaluation_outputs/comparative --suite ml --scheme RandomForest --seeds 2025

# 消融实验：单方案增量执行
python -m evaluation.run_ablation_experiments --config config.yaml --output-dir D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/evaluation_outputs/ablation --ablation-type Architecture --scheme Full --seeds 2025
python -m evaluation.run_ablation_experiments --config config.yaml --output-dir D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/evaluation_outputs/ablation --ablation-type Architecture --scheme NoPhysics --seeds 2025
python -m evaluation.run_ablation_experiments --config config.yaml --output-dir D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/evaluation_outputs/ablation --ablation-type Loss --scheme FullLoss --seeds 2025

# 核心四变体批量执行
python -m evaluation.run_core_ablation_suite --config config_ablation_quick.yaml --output-dir D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/evaluation_outputs/ablation_core4 --seeds 2025
```

如果需要继续沿用历史命令，也可以直接执行仓库根目录的兼容脚本：

```bash
python run_comparative_experiments.py ...
python run_ablation_experiments.py ...
python run_core_ablation_suite.py ...
```

## 关键参数

### 对比实验

- `--suite`：`physical`、`ml`、`deep`
- `--scheme`：指定单个模型方案；`auto` 时使用各组默认方案
- `--seeds`：支持一次传多个随机种子
- `--deep-opt-threshold`：仅对 AG-S2CNN 启用验证集阈值寻优
- `--deep-calibration`：`none`、`platt`、`isotonic`
- `--deep-tta`：启用测试时增强
- `--deep-boost`：启用训练强化包

### 消融实验

- `--ablation-type`：`Architecture`、`Loss`、`Window`、`HNM_Ratio`、`Margin`
- `--scheme`：指定当前只跑哪一个变体
- `--seeds`：控制重复实验种子
- `--device`：如 `cuda`、`cuda:0`、`cpu`

## 输出结果

### 对比实验输出

- `comparative_metrics_detail.csv`：每个 seed、每个模型的原始指标
- `comparative_metrics_summary.csv`：按模型汇总的 mean/std
- `comparative_leaderboard.csv`：按 F1、PR-AUC、MCC 综合排序
- `comparative_significance.csv`：AG-S2CNN 与其他模型的配对检验
- `comparative_tradeoff_scatter.png`
- `comparative_roc_curves.png`
- `comparative_pr_curves.png`
- `comparative_F1_bar.png`
- `comparative_PR_AUC_bar.png`
- `comparative_MCC_bar.png`
- `comparative_report_index.json`

### 消融实验输出

- `ablation_metrics_detail.csv`
- `ablation_metrics_summary.csv`
- `ablation_leaderboard.csv`
- `ablation_<type>_tradeoff_scatter.png`
- `ablation_<type>_F1_bar.png`
- `ablation_<type>_PR_AUC_bar.png`
- `ablation_<type>_MCC_bar.png`
- `ablation_report_index.json`

## 指标与统计

核心指标：

- `Precision`
- `Recall`
- `F1`
- `MCC`
- `Kappa`
- `ROC_AUC`
- `PR_AUC`
- `Brier`

解读建议：

- 非平衡场景优先看 `F1`、`PR_AUC`、`MCC`
- `ROC_AUC` 可辅助判断排序能力
- `Brier` 用于观察概率校准质量

显著性检验：

- 优先使用配对检验
- 不满足条件时回退到符号翻转置换检验

## 使用建议

- 对比实验建议按“单方案增量执行”方式跑完全部模型，便于中途查看阶段性结果。
- 消融实验建议先跑 `Architecture`，再跑 `Loss`，最后做参数敏感性。
- 若论文图重点聚焦核心贡献，优先运行 `run_core_ablation_suite.py`。
- 若需要新增图型或统计口径，优先扩展 `core.py`，不要把绘图逻辑散落到各入口脚本中。

## 结果解读

- 若 AG-S2CNN 在 `F1`、`PR_AUC`、`MCC` 上稳定领先，说明模型在目标检出与背景抑制之间取得更优平衡。
- 若 `NoPhysics`、`PhysicsConcat`、`PhysicsDiffNoAtt` 相比 `Full` 明显退化，可支撑物理先验与差分融合的必要性。
- 若 `StandardBCE` 相比 `FullLoss` 退化明显，可支撑增强损失设计的有效性。

## 备注

- 本模块输出逻辑已与 `Evaluation.txt` 的评价思路对齐。
- 如需继续生成论文级排版图、图注模板或章节化自动报告，可在此模块上继续扩展。
