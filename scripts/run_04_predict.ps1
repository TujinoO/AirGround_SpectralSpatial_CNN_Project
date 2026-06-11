$ErrorActionPreference = "Stop"
function Invoke-ProjectPython {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 @args
  } else {
    & python @args
  }
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location "$ProjectRoot\modules\03_ag_s2cnn_model"
Invoke-ProjectPython predict.py `
  --config "$ProjectRoot\configs\ag_s2cnn_example.yaml" `
  --checkpoint "$ProjectRoot\data\03_model\output\checkpoints\best_model.pth" `
  --predict-image "$ProjectRoot\data\04_predict\GF5A_AHSI_E89.3_N38.7_20240126_verification.tif" `
  --predict-output-dir "$ProjectRoot\data\04_predict\output"
Pop-Location
