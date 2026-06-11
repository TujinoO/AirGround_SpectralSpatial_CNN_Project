"""Allow the standalone evaluation module to reuse the AG-S2CNN model code."""
from pathlib import Path
import sys

MODEL_DIR = Path(__file__).resolve().parents[1] / "03_ag_s2cnn_model"
if MODEL_DIR.exists():
    model_path = str(MODEL_DIR)
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
