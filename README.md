# 空地协同光谱-空间卷积神经网络项目说明书

本项目是融合空地高光谱数据的深度学习模型在新疆塔木切矿集区伟晶岩锂矿预测中的应用研究工作的整理版，围绕“地面标准光谱库构建、遥感影像样本生产、AG-S2CNN 模型训练预测、模型评价与消融实验”形成一套完整的技术体系。项目已经把原来分散的三个核心代码模块和最新示例数据统一整理到一个大项目中，并额外提供了最简单的三条启动命令，方便后续复现、展示、归档和继续开发。


## 一、项目要解决什么问题

本项目面向高光谱遥感影像中的锂矿化目标识别任务，核心思想是把“地面实测光谱信息”和“卫星高光谱影像空间信息”结合起来，提高富矿伟晶岩等目标在复杂背景中的识别能力。

完整流程可以理解为三层递进：

1. 先用地面实测光谱构建标准参考光谱库，也就是 GSRSL。
2. 再用 GSRSL 和高光谱影像自动生成伪标签样本，作为深度学习训练监督信号。
3. 最后训练 AG-S2CNN 模型，用训练好的模型做预测，并进行精度评价、对比实验和消融实验。

对于日常使用，最常用的是三个工作模块：

1. 模型训练：使用已有示例数据训练或复训模型。
2. 模型预测：使用训练好的 checkpoint 对新影像进行预测出图。
3. 模型评价：快速评价当前模型，或者运行论文级对比/消融实验。

## 二、项目目录结构

```text
AirGround_SpectralSpatial_CNN_Project/
  README.md                         # 当前说明书
  requirements.txt                  # 总依赖清单
  train.cmd                         # 快捷命令：模型训练
  predict.cmd                       # 快捷命令：模型预测
  evaluate.cmd                      # 快捷命令：模型评价

  configs/
    workflow_settings.ps1           # 最简单的统一参数配置文件，优先改这里
    ag_s2cnn_example.yaml           # AG-S2CNN 训练/预测基础配置
    ag_s2cnn_ablation_quick.yaml    # 快速消融实验配置

  scripts/
    ag_train.ps1                    # train.cmd 调用的训练脚本
    ag_predict.ps1                  # predict.cmd 调用的预测脚本
    ag_evaluate.ps1                 # evaluate.cmd 调用的评价脚本
    workflow_common.ps1             # 三个快捷脚本共用的辅助函数
    run_01_build_gsrsl.ps1          # 完整流程第 1 步：构建 GSRSL
    run_02_generate_pseudo_labels.ps1 # 完整流程第 2 步：生成伪标签
    run_03_train_model.ps1          # 完整流程第 3 步：训练模型
    run_04_predict.ps1              # 完整流程第 4 步：预测
    run_05_core_ablation.ps1        # 完整流程第 5 步：核心消融实验

  modules/
    01_construction_of_gsrsl/        # 模块 1：地面标准参考光谱库构建
    02_ags2cnn_sample_production/   # 模块 2：伪标签样本生产
    03_ag_s2cnn_model/              # 模块 3：AG-S2CNN 模型训练与预测主体
    04_evaluation_experiments/      # 模块 4：模型评价、对比实验、消融实验

  data/
    01_gsrsl/                       # GSRSL 构建所需输入和输出示例
    02_sample_production/           # 伪标签生产所需输入和输出示例
    03_model/                       # 模型训练数据、标签、checkpoint、训练结果
    04_predict/                     # 预测影像和预测结果示例

  docs/
    QUICK_WORKFLOW.md               # 三条快捷命令的简明说明
    ENVIRONMENT_AND_VALIDATION.md   # 环境与验证记录
    ARCHIVE_MANIFEST.md             # 项目归档清单
    脚本交互清单.docx               # 原始脚本交互材料
```

## 三、最简单的使用方式

如果只是想用当前已经整理好的示例数据跑模型训练、预测和评价，只需要进入项目根目录，然后运行三条命令。

```powershell
cd E:\code\AirGround_SpectralSpatial_CNN_Project
.\train.cmd
.\predict.cmd
.\evaluate.cmd
```

这三条命令分别对应：

| 命令 | 作用 | 默认输入 | 默认输出 |
| --- | --- | --- | --- |
| `train.cmd` | 训练 AG-S2CNN 模型 | `data/03_model` 下的影像、伪标签和 GSRSL | `data/03_model/output` |
| `predict.cmd` | 使用 checkpoint 预测验证影像 | `data/04_predict/GF5A_AHSI_E89.3_N38.7_20240126_verification.tif` | `data/04_predict/output` |
| `evaluate.cmd` | 快速评价当前模型 | `data/03_model/output/checkpoints/best_model.pth` | 终端输出测试指标 |

第一次运行前建议先用 `-DryRun` 预览命令是否正确。`-DryRun` 不会真正训练或预测，只会打印即将执行的底层命令。

```powershell
.\train.cmd -DryRun
.\predict.cmd -DryRun
.\evaluate.cmd -DryRun
```

如果这三条预览命令能正常打印路径，说明快捷入口和默认配置是通的。

## 四、环境安装

项目使用 Python 运行。当前快捷脚本会优先使用 Windows 的 `py -3`，如果没有 `py -3`，再尝试使用 `python`。

建议在项目根目录安装依赖：

```powershell
cd E:\code\AirGround_SpectralSpatial_CNN_Project
py -3 -m pip install -r requirements.txt
```

如果你的电脑使用 GPU 训练，还需要确认 PyTorch 与本机 CUDA 版本匹配。若默认安装的 PyTorch 不能识别 GPU，需要到 PyTorch 官网选择对应 CUDA 版本重新安装。

主要依赖包括：

- `torch`、`torchvision`：深度学习训练和推理。
- `numpy`、`pandas`、`scipy`：数值计算和表格处理。
- `rasterio`：GeoTIFF 高光谱影像读取和地理参考保存。
- `scikit-learn`：传统机器学习对比、PCA/MNF 等处理。
- `matplotlib`、`seaborn`：结果图绘制。
- `pyyaml`：读取 YAML 配置文件。

## 五、最重要的参数配置文件

为了让使用尽量简单，项目新增了一个统一的傻瓜式配置文件：

```text
configs/workflow_settings.ps1
```

日常使用优先改这个文件，不需要每次手写复杂命令。常用参数如下。

### 1. Python 启动器

```powershell
$PythonLauncher = ""
```

保持空字符串即可，脚本会自动使用 `py -3` 或 `python`。如果你有指定环境，例如 Anaconda 环境里的 Python，可以改成完整路径：

```powershell
$PythonLauncher = "D:\Users\pytnon\python.exe"
```

### 2. 训练参数

```powershell
$TrainEpochs = $null
$TrainBatchSize = $null
$TrainLearningRate = $null
$TrainOutputDir = Join-Path $ProjectRoot "data\03_model\output"
$TrainGpuId = 0
$TrainSeed = 42
$TrainUseBoost = $false
```

说明：

- `$TrainEpochs`：训练轮数。为 `$null` 时使用 `configs/ag_s2cnn_example.yaml` 里的默认值。
- `$TrainBatchSize`：批大小。显存不够时可以改小，比如 `4` 或 `2`。
- `$TrainLearningRate`：学习率。常见值如 `0.001`、`0.0005`。
- `$TrainOutputDir`：训练结果输出目录。
- `$TrainGpuId`：GPU 编号，单卡通常为 `0`。
- `$TrainSeed`：随机种子，保证实验可复现。
- `$TrainUseBoost`：是否开启强化训练包。

例如想快速训练 20 轮，可以改成：

```powershell
$TrainEpochs = 20
$TrainBatchSize = 4
$TrainLearningRate = 0.0005
```

也可以不改配置文件，临时在命令后面覆盖：

```powershell
.\train.cmd -Epochs 20 -BatchSize 4 -LearningRate 0.0005
```

### 3. 预测参数

```powershell
$PredictCheckpoint = Join-Path $ProjectRoot "data\03_model\output\checkpoints\best_model.pth"
$PredictImage = Join-Path $ProjectRoot "data\04_predict\GF5A_AHSI_E89.3_N38.7_20240126_verification.tif"
$PredictOutputDir = Join-Path $ProjectRoot "data\04_predict\output"
$PredictGpuId = 0
$PredictUseTta = $false
$PredictProbabilityThreshold = $null
```

说明：

- `$PredictCheckpoint`：模型权重文件路径。
- `$PredictImage`：要预测的高光谱影像路径。
- `$PredictOutputDir`：预测结果保存目录。
- `$PredictGpuId`：GPU 编号。
- `$PredictUseTta`：是否启用测试时增强。
- `$PredictProbabilityThreshold`：二分类概率阈值，为 `$null` 时使用 YAML 配置默认值。

如果要换一张新影像预测，可以临时运行：

```powershell
.\predict.cmd -Image "E:\your_data\new_image.tif"
```

如果要换模型权重：

```powershell
.\predict.cmd -Checkpoint "E:\your_model\best_model.pth"
```

### 4. 评价参数

```powershell
$EvaluationMode = "test"
$EvaluationCheckpoint = $PredictCheckpoint
$EvaluationOutputDir = Join-Path $ProjectRoot "data\03_model\evaluation_outputs\ablation_core4_quick"
$EvaluationDevice = "cuda"
$EvaluationSeeds = @(2025)
$EvaluationGpuId = 0
$EvaluationUseTta = $false
```

评价有两种模式：

- `test`：快速评价当前 checkpoint，这是默认模式，适合日常检查模型精度。
- `ablation`：运行核心消融实验套件，适合论文对比和实验分析，耗时更长。

默认快速评价：

```powershell
.\evaluate.cmd
```

运行核心消融实验：

```powershell
.\evaluate.cmd -Mode ablation
```

或者在配置文件中改为：

```powershell
$EvaluationMode = "ablation"
```

## 六、三个工作模块的详细说明

### 工作模块 1：模型训练

快捷命令：

```powershell
.\train.cmd
```

底层调用：

```text
modules/03_ag_s2cnn_model/train.py
```

默认使用的主配置：

```text
configs/ag_s2cnn_example.yaml
```

默认训练输入：

```text
data/03_model/GF5A_AHSI_E89.3_N38.7_20240126_experiment.tif
data/03_model/pseudo_label_balance_refined_final.tif
data/03_model/gsrsl.npy
```

训练结束后，常见输出在：

```text
data/03_model/output/
```

包括：

- `checkpoints/best_model.pth`：最佳模型权重。
- `loss_curves.png`：训练和验证损失曲线。
- `learning_rate_curve.png`：学习率变化曲线。
- `confusion_matrix_test.png`：测试集混淆矩阵。
- `confusion_matrix_test_normalized.png`：归一化混淆矩阵。
- `class_performance_test.png`：类别指标图。
- `metrics_report.json`：训练和测试指标汇总。
- `prediction_map_dense.png`、`rich_ore_probability_map.png`、`ore_potential_overlay.png` 等预测可视化结果。

常见训练命令示例：

```powershell
# 使用默认配置训练
.\train.cmd

# 快速试跑 10 轮
.\train.cmd -Epochs 10

# 显存不足时减小 batch size
.\train.cmd -BatchSize 2

# 临时指定学习率
.\train.cmd -LearningRate 0.0005

# 使用强化训练包
.\train.cmd -Boost
```

### 工作模块 2：模型预测

快捷命令：

```powershell
.\predict.cmd
```

底层调用：

```text
modules/03_ag_s2cnn_model/predict.py
```

默认输入：

```text
模型权重：data/03_model/output/checkpoints/best_model.pth
预测影像：data/04_predict/GF5A_AHSI_E89.3_N38.7_20240126_verification.tif
```

默认输出目录：

```text
data/04_predict/output/
```

常见预测输出包括：

- `prediction_map_dense.tif`：预测类别图，保留 GeoTIFF 输出。
- `prediction_map_dense.png`：预测类别图 PNG 可视化。
- `prediction_confidence_map.tif`：预测置信度图。
- `rich_ore_probability_map.tif`：富矿概率图。
- `rich_ore_probability_map.png`：富矿概率图可视化。
- `feature_activation_map.tif`：特征激活图。
- `ore_potential_overlay.png`：矿化潜力叠加图。

常见预测命令示例：

```powershell
# 使用默认模型和默认验证影像预测
.\predict.cmd

# 换一张影像预测
.\predict.cmd -Image "E:\your_data\new_hyperspectral_image.tif"

# 换一个 checkpoint
.\predict.cmd -Checkpoint "E:\your_model\best_model.pth"

# 指定输出目录
.\predict.cmd -OutputDir "E:\your_output\predict_result"

# 启用 TTA
.\predict.cmd -Tta

# 指定概率阈值
.\predict.cmd -Threshold 0.6
```

### 工作模块 3：模型评价

快捷命令：

```powershell
.\evaluate.cmd
```

默认评价模式为 `test`，底层调用：

```text
modules/03_ag_s2cnn_model/train.py --mode test
```

它会加载当前 checkpoint，并基于配置中的数据划分进行快速测试评价。适合日常检查模型是否可用、checkpoint 是否匹配、测试指标是否正常。

默认 checkpoint：

```text
data/03_model/output/checkpoints/best_model.pth
```

常见评价命令示例：

```powershell
# 快速评价当前模型
.\evaluate.cmd

# 指定 checkpoint 评价
.\evaluate.cmd -Checkpoint "E:\your_model\best_model.pth"

# 启用 TTA 评价
.\evaluate.cmd -Tta
```

如果要运行论文级核心消融实验：

```powershell
.\evaluate.cmd -Mode ablation
```

消融实验底层调用：

```text
modules/04_evaluation_experiments/evaluation/run_core_ablation_suite.py
```

默认输出目录：

```text
data/03_model/evaluation_outputs/ablation_core4_quick/
```

可能生成：

- `ablation_metrics_detail.csv`：每个 seed、每个模型变体的详细指标。
- `ablation_metrics_summary.csv`：汇总指标。
- `ablation_leaderboard.csv`：指标排行榜。
- `ablation_Architecture_F1_bar.png`：F1 对比图。
- `ablation_Architecture_PR_AUC_bar.png`：PR-AUC 对比图。
- `ablation_Architecture_MCC_bar.png`：MCC 对比图。
- `ablation_Architecture_tradeoff_scatter.png`：指标权衡散点图。

## 七、完整技术流程脚本

除了日常三条命令之外，项目还保留了完整技术链路脚本。如果想从 GSRSL 构建开始完整复现，可以按顺序运行：

```powershell
cd E:\code\AirGround_SpectralSpatial_CNN_Project
.\scripts\run_01_build_gsrsl.ps1
.\scripts\run_02_generate_pseudo_labels.ps1
.\scripts\run_03_train_model.ps1
.\scripts\run_04_predict.ps1
.\scripts\run_05_core_ablation.ps1
```

这五个脚本对应：

| 步骤 | 脚本 | 作用 |
| --- | --- | --- |
| 1 | `run_01_build_gsrsl.ps1` | 根据地面光谱、标签和 GF-5 元数据构建 GSRSL |
| 2 | `run_02_generate_pseudo_labels.ps1` | 使用 GSRSL 和高光谱影像生成伪标签 |
| 3 | `run_03_train_model.ps1` | 训练 AG-S2CNN 模型 |
| 4 | `run_04_predict.ps1` | 使用模型进行全图预测 |
| 5 | `run_05_core_ablation.ps1` | 运行核心结构消融实验 |

日常使用建议优先用根目录三条快捷命令；完整流程脚本更适合复现实验链路或重新生成中间成果。

## 八、四个核心代码模块说明

### 1. `modules/01_construction_of_gsrsl`

功能：构建 Ground Standard Reference Spectral Library，即地面标准参考光谱库。

主要输入：

- 地面光谱矩阵 `spectra.csv`。
- 光谱类别标签 `labels.csv`。
- GF-5 AHSI 波段元数据 `GF5A_AHSI_20240126__metadata.txt`。

主要处理：

1. 读取地面实测光谱。
2. 使用 Savitzky-Golay 滤波进行光谱平滑。
3. 根据 GF-5 波段响应进行重采样。
4. 按岩性类别聚合平均光谱。
5. 输出 `gsrsl.npy` 和可视化图。

对应数据目录：

```text
data/01_gsrsl/
```

### 2. `modules/02_ags2cnn_sample_production`

功能：根据高光谱影像和 GSRSL 自动生成伪标签样本。

主要输入：

- GF-5 AHSI 高光谱影像。
- `gsrsl.npy` 标准光谱库。
- 波长元数据 `gf5a_wavelengths.csv`。

主要处理：

1. 使用 MNF / PCA 类方法提取主要光谱信息。
2. 使用 AMCS 进行分量筛选。
3. 使用 SID 光谱信息散度进行类别判别。
4. 生成富矿、贫矿、围岩等伪标签。
5. 输出 GeoTIFF 标签图和可视化图。

对应数据目录：

```text
data/02_sample_production/
```

### 3. `modules/03_ag_s2cnn_model`

功能：AG-S2CNN 模型训练、测试和预测。

主要组成：

- `train.py`：训练、验证、测试入口。
- `predict.py`：独立预测入口。
- `models/`：AG-S2CNN 网络结构。
- `utils/`：数据读取、损失函数、指标计算、日志和可视化工具。
- `config.py`：YAML 配置读取和验证。

对应数据目录：

```text
data/03_model/
data/04_predict/
```

### 4. `modules/04_evaluation_experiments`

功能：模型评价、对比实验和消融实验。

主要组成：

- `evaluation/run_comparative_experiments.py`：对比实验入口。
- `evaluation/run_ablation_experiments.py`：消融实验入口。
- `evaluation/run_core_ablation_suite.py`：核心四变体消融套件。
- `evaluation/core.py`：评价指标、数据准备、绘图、统计检验。
- `evaluation/models.py`：对比模型和消融模型构建。

该模块已经从模型主目录中独立拆分出来，但通过 `sitecustomize.py` 自动引用相邻的 `modules/03_ag_s2cnn_model`，因此不需要重复维护两份模型代码。

## 九、数据组织说明

当前项目只保留了最新活跃数据示例，没有打包历史数据目录。这样做的目的是让项目更清晰、更轻量，也避免误用旧实验数据。

数据目录含义：

```text
data/01_gsrsl
```

用于 GSRSL 构建，包含地面光谱、标签、元数据和 GSRSL 输出图。

```text
data/02_sample_production
```

用于伪标签生产，包含实验影像、波长文件、GSRSL、伪标签输出。

```text
data/03_model
```

用于模型训练和评价，包含训练影像、伪标签、GSRSL、checkpoint、训练结果、评价输出。

```text
data/04_predict
```

用于模型预测，包含验证影像和预测输出。

如果以后要换数据，建议保持同样的目录思想：

1. 训练数据放入 `data/03_model`。
2. 待预测影像放入 `data/04_predict`。
3. 修改 `configs/workflow_settings.ps1` 中的路径。
4. 先运行 `-DryRun` 检查路径。
5. 再正式运行训练、预测或评价。

## 十、常见问题

### 1. 运行时报 `ModuleNotFoundError: No module named 'torch'`

说明当前 Python 环境没有安装 PyTorch。先执行：

```powershell
py -3 -m pip install -r requirements.txt
```

如果仍然不行，说明你运行命令使用的 Python 环境和安装依赖的 Python 环境不是同一个。可以在 `configs/workflow_settings.ps1` 中指定 `$PythonLauncher` 为正确的 Python 路径。

### 2. CUDA 显存不足

可以减小 batch size：

```powershell
.\train.cmd -BatchSize 2
```

也可以在 `configs/workflow_settings.ps1` 中修改：

```powershell
$TrainBatchSize = 2
```

### 3. 想只检查命令，不想真正运行

使用 `-DryRun`：

```powershell
.\train.cmd -DryRun
.\predict.cmd -DryRun
.\evaluate.cmd -DryRun
```

### 4. 想换预测影像

```powershell
.\predict.cmd -Image "E:\your_data\new_image.tif"
```

或者修改：

```powershell
$PredictImage = "E:\your_data\new_image.tif"
```

### 5. 想换模型权重

```powershell
.\predict.cmd -Checkpoint "E:\your_model\best_model.pth"
.\evaluate.cmd -Checkpoint "E:\your_model\best_model.pth"
```

或者修改：

```powershell
$PredictCheckpoint = "E:\your_model\best_model.pth"
$EvaluationCheckpoint = $PredictCheckpoint
```

### 6. 预测输出为空或结果不合理

可以检查：

1. 影像波段数是否与模型配置一致，默认是 297 个波段。
2. checkpoint 是否来自同一套训练配置。
3. `gsrsl.npy` 是否与训练时使用的数据一致。
4. 概率阈值是否过高，可以尝试 `-Threshold 0.4` 或 `-Threshold 0.5`。

### 7. 评价模式太慢

默认 `evaluate.cmd` 是快速测试模式。如果你把 `$EvaluationMode` 改成了 `ablation`，会运行消融实验，耗时明显更长。想恢复快速评价，改回：

```powershell
$EvaluationMode = "test"
```

## 十一、建议的新手学习顺序

如果完全不了解这个项目，建议按下面顺序学习：

1. 先读本 README，了解项目整体结构。
2. 运行三条 `-DryRun`，理解三个快捷命令实际调用了什么。
3. 打开 `configs/workflow_settings.ps1`，熟悉最常改的路径和参数。
4. 运行 `predict.cmd`，先用已有 checkpoint 对验证影像出图。
5. 运行 `evaluate.cmd`，查看当前 checkpoint 的测试指标。
6. 有 GPU 和足够时间后，再运行 `train.cmd` 重新训练。
7. 最后再研究 `scripts/run_01_build_gsrsl.ps1` 到 `run_05_core_ablation.ps1` 的完整技术链路。

## 十二、项目归档说明

本项目整理自以下原始模块：

- `D:\Code\Construction_of_GSRSL`
- `D:\Code\AGS2-CNN_Sample_Production`
- `D:\Code\Air-Ground_Spectral-Spatial_CNN`

示例数据整理自：

- `D:\Grp_data\Air-Ground_Spectral-Spatial_CNN`

整理后的项目没有修改或删除原始源文件。历史数据目录没有复制进来，当前 `data/` 下保留的是最新活跃示例数据和成果。

更多归档和验证信息见：

- `docs/ARCHIVE_MANIFEST.md`
- `docs/ENVIRONMENT_AND_VALIDATION.md`
- `docs/QUICK_WORKFLOW.md`
