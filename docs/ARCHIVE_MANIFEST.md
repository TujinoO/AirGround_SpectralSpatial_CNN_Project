# Archive manifest

Project root: E:\code\AirGround_SpectralSpatial_CNN_Project
Created: 2026-06-11 14:43:08

## Included source modules

- `modules/01_construction_of_gsrsl`: GSRSL construction code copied from `D:\Code\Construction_of_GSRSL`.
- `modules/02_ags2cnn_sample_production`: pseudo-label sample production copied from `D:\Code\AGS2-CNN_Sample_Production`.
- `modules/03_ag_s2cnn_model`: AG-S2CNN training and prediction core copied from `D:\Code\Air-Ground_Spectral-Spatial_CNN`.
- `modules/04_evaluation_experiments`: comparative experiments, ablation experiments, and paper-figure post-processing split from the AG-S2CNN project.

## Included current example data

Current data was copied from top-level folders under `D:\Grp_data\Air-Ground_Spectral-Spatial_CNN`:

- `data/01_gsrsl`
- `data/02_sample_production`
- `data/03_model`
- `data/04_predict`

`historical_data` was intentionally not copied, so the packaged examples use the latest active dataset rather than earlier runs.

## Cleanup policy

Excluded generated/cache material:

- `__pycache__`
- `.pytest_cache`, `.mypy_cache`, `.ruff_cache`
- `gsrsl_pipeline.egg-info`
- AG-S2CNN root `logs/` and `results/`
- duplicate root compatibility experiment entry scripts from the AG-S2CNN model module

No original source or data files were modified or deleted.

