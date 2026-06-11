param(
    [string]$Config = "",
    [int]$Epochs = 0,
    [int]$BatchSize = 0,
    [double]$LearningRate = 0,
    [string]$OutputDir = "",
    [int]$Gpu = -999,
    [int]$Seed = -1,
    [switch]$Boost,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
. "$ProjectRoot\configs\workflow_settings.ps1"
. "$ProjectRoot\scripts\workflow_common.ps1"

$modelDir = Join-Path $ProjectRoot "modules\03_ag_s2cnn_model"
$configPath = if ($Config) { $Config } else { $ModelConfig }
$outDir = if ($OutputDir) { $OutputDir } else { $TrainOutputDir }
$gpuId = if ($Gpu -ne -999) { $Gpu } else { $TrainGpuId }
$seedValue = if ($Seed -ge 0) { $Seed } else { $TrainSeed }

Assert-PathExists $configPath "Model config"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$cmdArgs = [System.Collections.Generic.List[string]]::new()
$cmdArgs.Add("train.py") | Out-Null
$cmdArgs.Add("--config") | Out-Null
$cmdArgs.Add($configPath) | Out-Null
$cmdArgs.Add("--mode") | Out-Null
$cmdArgs.Add("train") | Out-Null
Add-OptionalArg -Args $cmdArgs -Name "--output-dir" -Value $outDir
Add-OptionalArg -Args $cmdArgs -Name "--gpu" -Value $gpuId
Add-OptionalArg -Args $cmdArgs -Name "--seed" -Value $seedValue
if ($Epochs -gt 0) { Add-OptionalArg -Args $cmdArgs -Name "--epochs" -Value $Epochs } elseif ($null -ne $TrainEpochs) { Add-OptionalArg -Args $cmdArgs -Name "--epochs" -Value $TrainEpochs }
if ($BatchSize -gt 0) { Add-OptionalArg -Args $cmdArgs -Name "--batch-size" -Value $BatchSize } elseif ($null -ne $TrainBatchSize) { Add-OptionalArg -Args $cmdArgs -Name "--batch-size" -Value $TrainBatchSize }
if ($LearningRate -gt 0) { Add-OptionalArg -Args $cmdArgs -Name "--lr" -Value $LearningRate } elseif ($null -ne $TrainLearningRate) { Add-OptionalArg -Args $cmdArgs -Name "--lr" -Value $TrainLearningRate }
if ($Boost -or $TrainUseBoost) { $cmdArgs.Add("--boost") | Out-Null }

Write-Host "[TRAIN] Config: $configPath"
Write-Host "[TRAIN] Output: $outDir"
if ($DryRun) {
    Write-Host "[DRY-RUN] Working directory: $modelDir"
    Write-Host "[DRY-RUN] Command: $(Get-ProjectPythonCommand -ConfiguredLauncher $PythonLauncher) $($cmdArgs -join ' ')"
    return
}
Push-Location $modelDir
try {
    Invoke-ProjectPython -Arguments $cmdArgs.ToArray() -Launcher $PythonLauncher
} finally {
    Pop-Location
}
