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
Invoke-ProjectPython train.py --config "$ProjectRoot\configs\ag_s2cnn_example.yaml" --mode train
Pop-Location
