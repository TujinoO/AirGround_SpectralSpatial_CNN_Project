# Environment and validation notes

## Recommended environment setup

On this machine, the Windows `py -3` launcher is available, while the plain `python` command points to an environment that exits unexpectedly. The project PowerShell scripts therefore prefer `py -3` and fall back to `python` only when `py` is unavailable.

Install dependencies before running the full workflow:

```powershell
cd "E:\code\AirGround_SpectralSpatial_CNN_Project"
py -3 -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching the local CUDA version if the default pip package is not appropriate.

## Validation performed

- Copied project contains 381 files, about 1.04 GB.
- No `historical_data`, `__pycache__`, `.pytest_cache`, or `.mypy_cache` directories were found in the organized project.
- Parsed 128 Python files with `ast.parse`; all passed syntax validation.
- Verified key example paths exist under the new project:
  - `data/01_gsrsl/GF5A_AHSI_20240126__metadata.txt`
  - `data/01_gsrsl/spectra.csv`
  - `data/01_gsrsl/labels.csv`
  - `data/02_sample_production/GF5A_AHSI_E89.3_N38.7_20240126_experiment.tif`
  - `data/02_sample_production/gf5a_wavelengths.csv`
  - `data/03_model/GF5A_AHSI_E89.3_N38.7_20240126_experiment.tif`
  - `data/03_model/pseudo_label_balance_refined_final.tif`
  - `data/03_model/gsrsl.npy`
  - `data/03_model/output/checkpoints/best_model.pth`
  - `data/04_predict/GF5A_AHSI_E89.3_N38.7_20240126_verification.tif`
- Confirmed `configs/ag_s2cnn_example.yaml` and `configs/ag_s2cnn_ablation_quick.yaml` point to the copied `E:\code` project data, not the original `D:\Grp_data` directory.
- Confirmed `modules/03_ag_s2cnn_model` no longer contains the root comparative/ablation experiment scripts; evaluation code is split into `modules/04_evaluation_experiments`.

## Entry checks

Passed:

```powershell
py -3 build_gsrsl.py --help
py -3 -m evaluation.run_core_ablation_suite --help
```

Blocked by missing dependencies in the current `py -3` environment:

```text
py -3 -m hyperspectral_pseudo_label_generator.cli --help
ModuleNotFoundError: No module named 'sklearn'

py -3 train.py --help
ModuleNotFoundError: No module named 'torch'

PyYAML config parsing check
ModuleNotFoundError: No module named 'yaml'
```

These are environment dependency gaps. The required packages are listed in the root `requirements.txt`.

