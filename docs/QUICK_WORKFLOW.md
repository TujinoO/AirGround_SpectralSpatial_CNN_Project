# Quick three-command workflow

This project now has three root-level commands for the three daily work modules.

Run from PowerShell or Command Prompt:

```powershell
cd "E:\code\AirGround_SpectralSpatial_CNN_Project"
.\train.cmd
.\predict.cmd
.\evaluate.cmd
```

## What each command does

- `train.cmd`: trains AG-S2CNN with the default example data in `data/03_model`.
- `predict.cmd`: loads `data/03_model/output/checkpoints/best_model.pth` and predicts the default verification image in `data/04_predict`.
- `evaluate.cmd`: by default, quickly evaluates the current checkpoint on the configured test split.

## One simple place to edit parameters

Edit:

```text
configs/workflow_settings.ps1
```

Common changes:

```powershell
$TrainEpochs = 30
$TrainBatchSize = 4
$TrainLearningRate = 0.0005
$PredictCheckpoint = Join-Path $ProjectRoot "data\03_model\output\checkpoints\best_model.pth"
$PredictImage = Join-Path $ProjectRoot "data\04_predict\GF5A_AHSI_E89.3_N38.7_20240126_verification.tif"
$EvaluationMode = "test"
```

For thesis-level core ablation evaluation, change:

```powershell
$EvaluationMode = "ablation"
$EvaluationDevice = "cuda"
$EvaluationSeeds = @(2025)
```

## Optional one-time overrides

You can override important settings without editing the config file:

```powershell
.\train.cmd -Epochs 20 -BatchSize 4
.\predict.cmd -Image "E:\path\to\new_image.tif"
.\evaluate.cmd -Mode ablation -Device cuda -Seeds 2025
```

The root commands use `py -3` automatically when available, then fall back to `python`.

## Preview commands without running

```powershell
.\train.cmd -DryRun
.\predict.cmd -DryRun
.\evaluate.cmd -DryRun
```
