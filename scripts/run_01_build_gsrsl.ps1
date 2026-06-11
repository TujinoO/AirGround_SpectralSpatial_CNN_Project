$ErrorActionPreference = "Stop"
function Invoke-ProjectPython {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 @args
  } else {
    & python @args
  }
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location "$ProjectRoot\modules\01_construction_of_gsrsl"
Invoke-ProjectPython build_gsrsl.py `
  --metadata "$ProjectRoot\data\01_gsrsl\GF5A_AHSI_20240126__metadata.txt" `
  --spectra "$ProjectRoot\data\01_gsrsl\spectra.csv" `
  --labels "$ProjectRoot\data\01_gsrsl\labels.csv" `
  --output "$ProjectRoot\data\01_gsrsl\output"
Pop-Location
