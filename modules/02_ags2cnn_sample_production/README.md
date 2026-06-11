# Hyperspectral Pseudo-Label Generator
# 高光谱伪标签生成器

## 概述

本项目用于 GF-5 AHSI 等 297 波段高光谱影像的高置信度伪标签生成。当前主线方案已从“PCA 方差驱动”升级为“**MNF 信噪比驱动 + AMCS 自动分量优选 + 选择性逆变换 + SID 严格判别**”。

目标是获得可直接用于后续深度学习训练的高纯度监督信号，并尽量抑制地形阴影、云影与高频噪声造成的误标。

---

## 技术路线（与样本工程260319一致）

### 1) MNF 两步变换（SNR 驱动）

- 第一步：噪声协方差估计与白化（Noise Whitening）
- 第二步：在白化空间做特征分解，得到按质量递减的分量

实现要点：
- 使用空间差分估计噪声样本并构建噪声协方差
- 特征值分解后得到白化矩阵与反白化矩阵

### 2) AMCS 自动分量优选（斩头去尾）

- **斩头（阴影剥离）**：分量与全波段均值图（全色等效图）做 Pearson 相关，`|R_i|` 高于阈值则剔除
- **去尾（噪声剥离）**：分量同时满足“特征值低（近噪声）+ Moran’s I 低（空间无序）”则剔除
- 若有效分量不足，按特征值由高到低补齐到 `n_components`

### 3) 选择性逆变换重构

- 仅保留 AMCS 选中的分量，其余置零
- 执行逆变换回到 297 波段物理反射率空间
- 用重构后的影像与重构后的参考光谱进入 SID

### 4) SID 与四类严格标注

- 先计算 5 类参考的 SID
- 聚合为 3 类判别量：
  - `rich = min(SID_0, SID_1, SID_2)`
  - `poor = SID_3`
  - `wall = SID_4`
- 分类输出四类：
  - `0` 未标注/歧义
  - `1` 富矿伟晶岩（Rich Ore Pegmatite）
  - `2` 贫矿伟晶岩（Poor Ore Pegmatite）
  - `3` 围岩（Wall Rock）

### 5) 论文级可视化输出

- 可视化图仅展示 `1/2` 两类矿化标签（富矿/贫矿），不显示围岩标签
- 伪标签主输出 `.tif/.npy` 仍完整保留四类编码（含围岩）
- 可视化坐标轴优先读取输入 GeoTIFF 地理参考并转换为经纬度坐标（EPSG:4326）
- 统一高分辨率导出（600 DPI），可直接用于论文制图

---

## 输入要求

### 高光谱影像 `--image`
- 支持：`.tif/.tiff`、`.npy`
- 形状：`(H, W, 297)`
- 建议：GeoTIFF（可直接保留地理参考用于输出）

### 参考光谱 `--reference`
- 支持：`.npy`、`.csv`
- 语义类别：
  - `0` 锂辉石型富矿伟晶岩
  - `1` 锂云母型富矿伟晶岩
  - `2` 混合型富矿伟晶岩
  - `3` 贫矿伟晶岩
  - `4` 围岩

### 波长元数据 `--metadata`
- 支持：`.json`、`.csv`、GF5A 风格 `.txt`
- 必须包含 297 个波长值

---

## 输出结果

主输出与可视化如下：

1. `pseudo_label_map*.npy` 或 `pseudo_label_map*.tif`
- 伪标签图，`(H, W)`，`uint8`
- 若输出 `.tif`，自动复制输入 GeoTIFF 的地理参考（CRS/Transform）

2. `pseudo_label_visualization.png`
- 仅富矿/贫矿两类的空间分布图 + 统计柱状图
- 带真实地理坐标轴（优先经纬度）

3. `pseudo_label_overlay_rgb.png`
- 灰度底图叠加富矿/贫矿标签融合图（文件名沿用历史命名）
- 带真实地理坐标轴与图例，适合直接展示空间位置关系

---

## 关键参数

### 主流程参数
- `--n-components`：AMCS 最终保留分量数
- `--ore-percentile`：富矿阈值百分位
- `--non-ore-percentile`：非富矿阈值百分位（用于贫矿/围岩判别）
- `--ambiguity-threshold`：歧义抑制阈值
- `--shadow-exclusion-percentile`：分类后阴影剔除百分位（推荐 `5.0`，0 表示关闭）
- `--chunk-size`：SID 分块计算块大小

### AMCS 参数
- `--amcs-shadow-corr-threshold`：阴影相关阈值（推荐 `0.14`）
- `--amcs-snr-threshold`：低信噪比判定阈值（推荐 `1.8`）
- `--amcs-moran-threshold`：空间无序阈值（推荐 `0.10`）

---

## 运行示例（你当前测试数据）

```powershell
python -m hyperspectral_pseudo_label_generator.cli --image D:\Grp_data\Air-Ground_Spectral-Spatial_CNN\2_Sample_Production\GF5A_AHSI_E89.3_N38.7_20240126_experiment.tif --reference "D:\Grp_data\Air-Ground_Spectral-Spatial_CNN\2_Sample_Production\gsrsl.npy" `
  --metadata "D:\Grp_data\Air-Ground_Spectral-Spatial_CNN\2_Sample_Production\gf5a_wavelengths.csv" `
  --n-components 10 `
  --ore-percentile 17.0 `
  --non-ore-percentile 6.5 `
  --ambiguity-threshold 2e-7 `
  --chunk-size 1000 `
  --amcs-shadow-corr-threshold 0.14 `
  --amcs-snr-threshold 1.8 `
  --amcs-moran-threshold 0.10 `
  --sid-smoothing-window 3 `
  --sid-invariant-weight 0.6 `
  --sid-disagreement-penalty 0.6 `
  --pre-shadow-global-percentile 25.0 `
  --pre-shadow-local-percentile 25.0 `
  --pre-edge-percentile 90.0 `
  --pre-unreliable-dilation-radius 2 `
  --shadow-exclusion-percentile 22.0 `
  --shadow-local-percentile 22.0 `
  --shadow-local-window-size 151 `
  --edge-exclusion-percentile 88.0 `
  --rich-min-neighbors 1 `
  --poor-min-neighbors 1 `
  --rich-core-window-size 25 `
  --rich-core-min-density 0.015 `
  --poor-near-rich-radius 12 `
  --rich-to-poor-sid-margin -3e-4 `
  --poor-confidence-percentile 2.0 `
  --output-dir "D:\Grp_data\Air-Ground_Spectral-Spatial_CNN\2_Sample_Production\output_Presentation" `
  --output-file "pseudo_label_balance_refined_final.tif"
```

---

## Python 调用示例

```python
from hyperspectral_pseudo_label_generator import ProcessingConfig, PseudoLabelPipeline

config = ProcessingConfig(
    n_components=10,
    ore_percentile=17.0,
    non_ore_percentile=6.5,
    ambiguity_threshold=2e-7,
    chunk_size=1000,
    amcs_shadow_corr_threshold=0.14,
    amcs_snr_threshold=1.8,
    amcs_moran_threshold=0.10,
    sid_smoothing_window=3,
    sid_invariant_weight=0.6,
    sid_disagreement_penalty=0.6,
    pre_shadow_global_percentile=25.0,
    pre_shadow_local_percentile=25.0,
    pre_edge_percentile=90.0,
    pre_unreliable_dilation_radius=2,
    shadow_exclusion_percentile=22.0,
    shadow_local_percentile=22.0,
    shadow_local_window_size=151,
    edge_exclusion_percentile=88.0,
    rich_min_neighbors=1,
    poor_min_neighbors=1,
    rich_core_window_size=25,
    rich_core_min_density=0.015,
    poor_near_rich_radius=12,
    rich_to_poor_sid_margin=-3e-4,
    poor_confidence_percentile=2.0,
    output_dir=r"D:\Grp_data\Air-Ground_Spectral-Spatial_CNN\2_Sample_Production\output",
)

pipeline = PseudoLabelPipeline(config)
labels = pipeline.run(
    image_path=r"D:\Grp_data\Air-Ground_Spectral-Spatial_CNN\2_Sample_Production\GF5A_AHSI_E89.3_N38.7_20240126_experiment.tif",
    reference_path=r"D:\Grp_data\Air-Ground_Spectral-Spatial_CNN\2_Sample_Production\gsrsl.npy",
    metadata_path=r"D:\Grp_data\Air-Ground_Spectral-Spatial_CNN\2_Sample_Production\gf5a_wavelengths.csv",
    output_filename="pseudo_label_balance_refined_final.tif",
)
```

---

## 代码结构

```text
hyperspectral_pseudo_label_generator/
  cli.py
  config.py
  pipeline.py
  input/
    image_loader.py
    reference_loader.py
    metadata_parser.py
    validator.py
  pca/
    physical_guided_pca.py
  sid/
    sid_calculator.py
  classification/
    thresholder.py
    classifier.py
  output/
    serializer.py
    visualizer.py
    statistics.py
```

---

## 验证命令

```bash
python -m pytest -q
python -m flake8
python -m mypy hyperspectral_pseudo_label_generator
```

---

## 依赖

- numpy
- scikit-learn
- rasterio
- matplotlib
- pytest / hypothesis
