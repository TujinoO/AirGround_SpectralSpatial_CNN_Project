param(
    [string]$Config = "",
    [string]$Checkpoint = "",
    [string]$Image = "",
    [string]$OutputDir = "",
    [int]$Gpu = -999,
    [double]$Threshold = -1,
    [switch]$Tta,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
. "$ProjectRoot\configs\workflow_settings.ps1"
. "$ProjectRoot\scripts\workflow_common.ps1"

$modelDir = Join-Path $ProjectRoot "modules\03_ag_s2cnn_model"
$configPath = if ($Config) { $Config } else { $ModelConfig }
$checkpointPath = if ($Checkpoint) { $Checkpoint } else { $PredictCheckpoint }
$imagePath = if ($Image) { $Image } else { $PredictImage }
$outDir = if ($OutputDir) { $OutputDir } else { $PredictOutputDir }
$gpuId = if ($Gpu -ne -999) { $Gpu } else { $PredictGpuId }
$thresholdValue = if ($Threshold -ge 0) { $Threshold } else { $PredictProbabilityThreshold }

Assert-PathExists $configPath "Model config"
Assert-PathExists $checkpointPath "Checkpoint"
Assert-PathExists $imagePath "Prediction image"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$cmdArgs = [System.Collections.Generic.List[string]]::new()
$cmdArgs.Add("predict.py") | Out-Null
$cmdArgs.Add("--config") | Out-Null
$cmdArgs.Add($configPath) | Out-Null
$cmdArgs.Add("--checkpoint") | Out-Null
$cmdArgs.Add($checkpointPath) | Out-Null
$cmdArgs.Add("--predict-image") | Out-Null
$cmdArgs.Add($imagePath) | Out-Null
$cmdArgs.Add("--predict-output-dir") | Out-Null
$cmdArgs.Add($outDir) | Out-Null
Add-OptionalArg -Args $cmdArgs -Name "--gpu" -Value $gpuId
if ($null -ne $thresholdValue) { Add-OptionalArg -Args $cmdArgs -Name "--prob-threshold" -Value $thresholdValue }
if ($Tta -or $PredictUseTta) { $cmdArgs.Add("--tta") | Out-Null }

Write-Host "[PREDICT] Image: $imagePath"
Write-Host "[PREDICT] Checkpoint: $checkpointPath"
Write-Host "[PREDICT] Output: $outDir"
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
