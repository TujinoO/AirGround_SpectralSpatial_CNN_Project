param(
    [ValidateSet("test", "ablation")][string]$Mode = "",
    [string]$Config = "",
    [string]$Checkpoint = "",
    [string]$OutputDir = "",
    [string]$Device = "",
    [int[]]$Seeds = @(),
    [int]$Gpu = -999,
    [switch]$Tta,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
. "$ProjectRoot\configs\workflow_settings.ps1"
. "$ProjectRoot\scripts\workflow_common.ps1"

$modeValue = if ($Mode) { $Mode } else { $EvaluationMode }

if ($modeValue -eq "test") {
    $modelDir = Join-Path $ProjectRoot "modules\03_ag_s2cnn_model"
    $configPath = if ($Config) { $Config } else { $ModelConfig }
    $checkpointPath = if ($Checkpoint) { $Checkpoint } else { $EvaluationCheckpoint }
    $gpuId = if ($Gpu -ne -999) { $Gpu } else { $EvaluationGpuId }

    Assert-PathExists $configPath "Model config"
    Assert-PathExists $checkpointPath "Checkpoint"

    $cmdArgs = [System.Collections.Generic.List[string]]::new()
    $cmdArgs.Add("train.py") | Out-Null
    $cmdArgs.Add("--config") | Out-Null
    $cmdArgs.Add($configPath) | Out-Null
    $cmdArgs.Add("--mode") | Out-Null
    $cmdArgs.Add("test") | Out-Null
    $cmdArgs.Add("--checkpoint") | Out-Null
    $cmdArgs.Add($checkpointPath) | Out-Null
    Add-OptionalArg -Args $cmdArgs -Name "--gpu" -Value $gpuId
    if ($Tta -or $EvaluationUseTta) { $cmdArgs.Add("--tta") | Out-Null }

    Write-Host "[EVALUATE:test] Config: $configPath"
    Write-Host "[EVALUATE:test] Checkpoint: $checkpointPath"
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
} else {
    $evalDir = Join-Path $ProjectRoot "modules\04_evaluation_experiments"
    $configPath = if ($Config) { $Config } else { $AblationConfig }
    $outDir = if ($OutputDir) { $OutputDir } else { $EvaluationOutputDir }
    $deviceValue = if ($Device) { $Device } else { $EvaluationDevice }
    $seedValues = if ($Seeds.Count -gt 0) { $Seeds } else { $EvaluationSeeds }

    Assert-PathExists $configPath "Ablation config"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null

    $cmdArgs = [System.Collections.Generic.List[string]]::new()
    $cmdArgs.Add("-m") | Out-Null
    $cmdArgs.Add("evaluation.run_core_ablation_suite") | Out-Null
    $cmdArgs.Add("--config") | Out-Null
    $cmdArgs.Add($configPath) | Out-Null
    $cmdArgs.Add("--output-dir") | Out-Null
    $cmdArgs.Add($outDir) | Out-Null
    $cmdArgs.Add("--device") | Out-Null
    $cmdArgs.Add($deviceValue) | Out-Null
    $cmdArgs.Add("--seeds") | Out-Null
    foreach ($seed in $seedValues) { $cmdArgs.Add([string]$seed) | Out-Null }

    Write-Host "[EVALUATE:ablation] Config: $configPath"
    Write-Host "[EVALUATE:ablation] Output: $outDir"
    Write-Host "[EVALUATE:ablation] Device: $deviceValue"
    if ($DryRun) {
        Write-Host "[DRY-RUN] Working directory: $evalDir"
        Write-Host "[DRY-RUN] Command: $(Get-ProjectPythonCommand -ConfiguredLauncher $PythonLauncher) $($cmdArgs -join ' ')"
        return
    }
    Push-Location $evalDir
    try {
        Invoke-ProjectPython -Arguments $cmdArgs.ToArray() -Launcher $PythonLauncher
    } finally {
        Pop-Location
    }
}
