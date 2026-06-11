$ErrorActionPreference = "Stop"
function Invoke-ProjectPython {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 @args
  } else {
    & python @args
  }
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location "$ProjectRoot\modules\02_ags2cnn_sample_production"
Invoke-ProjectPython -m hyperspectral_pseudo_label_generator.cli `
  --image "$ProjectRoot\data\02_sample_production\GF5A_AHSI_E89.3_N38.7_20240126_experiment.tif" `
  --reference "$ProjectRoot\data\02_sample_production\gsrsl.npy" `
  --metadata "$ProjectRoot\data\02_sample_production\gf5a_wavelengths.csv" `
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
  --output-dir "$ProjectRoot\data\02_sample_production\output" `
  --output-file "pseudo_label_balance_refined_final.tif"
Pop-Location
