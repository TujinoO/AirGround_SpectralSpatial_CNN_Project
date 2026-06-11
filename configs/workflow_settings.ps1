# Simple workflow settings for training, prediction, and evaluation.
# Most users only need to edit this file.

$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Python launcher. Leave empty to auto-use `py -3`, then `python`.
$PythonLauncher = ""

# Shared model configuration.
$ModelConfig = Join-Path $ProjectRoot "configs\ag_s2cnn_example.yaml"
$AblationConfig = Join-Path $ProjectRoot "configs\ag_s2cnn_ablation_quick.yaml"

# Training defaults. Use $null to keep the value inside ModelConfig.
$TrainEpochs = $null
$TrainBatchSize = $null
$TrainLearningRate = $null
$TrainOutputDir = Join-Path $ProjectRoot "data\03_model\output"
$TrainGpuId = 0
$TrainSeed = 42
$TrainUseBoost = $false

# Prediction defaults.
$PredictCheckpoint = Join-Path $ProjectRoot "data\03_model\output\checkpoints\best_model.pth"
$PredictImage = Join-Path $ProjectRoot "data\04_predict\GF5A_AHSI_E89.3_N38.7_20240126_verification.tif"
$PredictOutputDir = Join-Path $ProjectRoot "data\04_predict\output"
$PredictGpuId = 0
$PredictUseTta = $false
$PredictProbabilityThreshold = $null

# Evaluation defaults.
# "test" quickly evaluates the current checkpoint on the configured test split.
# "ablation" runs the core ablation experiment suite for thesis-level comparison.
$EvaluationMode = "test"
$EvaluationCheckpoint = $PredictCheckpoint
$EvaluationOutputDir = Join-Path $ProjectRoot "data\03_model\evaluation_outputs\ablation_core4_quick"
$EvaluationDevice = "cuda"
$EvaluationSeeds = @(2025)
$EvaluationGpuId = 0
$EvaluationUseTta = $false
