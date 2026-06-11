$ErrorActionPreference = "Stop"
function Invoke-ProjectPython {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 @args
  } else {
    & python @args
  }
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location "$ProjectRoot\modules\04_evaluation_experiments"
Invoke-ProjectPython -m evaluation.run_core_ablation_suite `
  --config "$ProjectRoot\configs\ag_s2cnn_ablation_quick.yaml" `
  --output-dir "$ProjectRoot\data\03_model\evaluation_outputs\ablation_core4_quick" `
  --seeds 2025
Pop-Location
