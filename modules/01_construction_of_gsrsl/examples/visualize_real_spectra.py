import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gsrsl_pipeline.data_models import GroundSpectrum
from gsrsl_pipeline.filters import apply_savgol_filter


CLASS_NAMES: Dict[int, str] = {
    0: "Spodumene-rich",
    1: "Lepidolite-rich",
    2: "Mixed-type",
    3: "Barren",
    4: "Wall Rock",
}


def load_spectra_matrix(spectra_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(spectra_path)
    if df.shape[1] < 2:
        raise ValueError(
            f"Spectrum matrix file '{spectra_path}' must have at least two columns."
        )
    wavelengths = df.iloc[:, 0].to_numpy(dtype=np.float64)
    reflectance_df = df.iloc[:, 1:].copy()
    reflectance_df.columns = reflectance_df.columns.astype(str)
    return wavelengths, reflectance_df


def load_labels(labels_path: Path, available_columns: List[str]) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    if "filename" not in labels.columns or "class_id" not in labels.columns:
        raise ValueError(
            f"Label table must contain 'filename' and 'class_id' columns. "
            f"Found columns: {list(labels.columns)}"
        )
    labels = labels.dropna(subset=["class_id"])
    if labels.empty:
        raise ValueError("Label table contains no class_id values.")
    labels["filename"] = labels["filename"].astype(str)
    labels["class_id"] = labels["class_id"].astype(int)
    labels = labels[labels["filename"].isin(available_columns)]
    if labels.empty:
        raise ValueError(
            "No labeled spectra match the columns in spectra.csv. "
            "Please ensure filenames in labels.csv correspond to column names in spectra.csv."
        )
    return labels


def compute_class_means(
    wavelengths: np.ndarray,
    reflectance_df: pd.DataFrame,
    labels: pd.DataFrame,
) -> Dict[int, GroundSpectrum]:
    class_means: Dict[int, GroundSpectrum] = {}
    for class_id in sorted(labels["class_id"].unique()):
        class_labels = labels[labels["class_id"] == class_id]
        col_names = class_labels["filename"].tolist()
        cols_present = [c for c in col_names if c in reflectance_df.columns]
        if not cols_present:
            continue
        values = reflectance_df[cols_present].to_numpy(dtype=np.float64)
        mean_reflectance = np.nanmean(values, axis=1)
        mean_reflectance = np.clip(mean_reflectance, 0.0, 1.0)
        spectrum = GroundSpectrum(
            wavelengths=wavelengths,
            reflectances=mean_reflectance,
            class_id=int(class_id),
            filename=f"class_{class_id}_mean",
            is_anomalous=False,
        )
        smoothed = apply_savgol_filter(spectrum)
        class_means[int(class_id)] = smoothed
    if not class_means:
        raise ValueError("No class means could be computed from the provided data.")
    return class_means


def select_example_spectra(
    wavelengths: np.ndarray,
    reflectance_df: pd.DataFrame,
    labels: pd.DataFrame,
) -> Dict[int, GroundSpectrum]:
    examples: Dict[int, GroundSpectrum] = {}
    for class_id in sorted(labels["class_id"].unique()):
        if class_id in examples:
            continue
        class_labels = labels[labels["class_id"] == class_id]
        for filename in class_labels["filename"]:
            if filename in reflectance_df.columns:
                reflectances = reflectance_df[filename].to_numpy(dtype=np.float64)
                reflectances = np.clip(reflectances, 0.0, 1.0)
                spectrum = GroundSpectrum(
                    wavelengths=wavelengths,
                    reflectances=reflectances,
                    class_id=int(class_id),
                    filename=str(filename),
                    is_anomalous=False,
                )
                examples[int(class_id)] = spectrum
                break
    if not examples:
        raise ValueError("No example spectra could be selected from the provided data.")
    return examples


def plot_class_means(
    class_means: Dict[int, GroundSpectrum],
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for idx, (class_id, spectrum) in enumerate(sorted(class_means.items())):
        name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        color = colors[idx % len(colors)]
        ax.plot(
            spectrum.wavelengths,
            spectrum.reflectances,
            label=f"{name} (n={spectrum.reflectances.size})",
            linewidth=1.5,
            color=color,
        )
    ax.set_xlabel("Wavelength / nm", fontsize=11)
    ax.set_ylabel("Reflectance", fontsize=11)
    ax.set_title("Mean Ground Spectra by Lithology (Smoothed)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(350, 2500)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)
    plt.tight_layout()
    output_path = output_dir / "real_spectra_class_means.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_example_spectra(
    examples: Dict[int, GroundSpectrum],
    output_dir: Path,
) -> Path:
    n_classes = len(examples)
    fig_height = 2.2 * n_classes
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, fig_height), sharex=True)
    if n_classes == 1:
        axes = [axes]
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for idx, (class_id, spectrum) in enumerate(sorted(examples.items())):
        ax = axes[idx]
        color = colors[idx % len(colors)]
        smoothed = apply_savgol_filter(spectrum)
        name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        ax.plot(
            spectrum.wavelengths,
            spectrum.reflectances,
            color="0.7",
            linewidth=0.6,
            label="Original",
        )
        ax.plot(
            smoothed.wavelengths,
            smoothed.reflectances,
            color=color,
            linewidth=1.3,
            label="Smoothed",
        )
        ax.set_ylabel("Reflectance", fontsize=10)
        ax.set_title(
            f"{name} – Example {spectrum.filename}",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Wavelength / nm", fontsize=11)
    plt.tight_layout()
    output_path = output_dir / "real_spectra_examples_smoothed.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize real ground spectra from spectra.csv and labels.csv.",
    )
    parser.add_argument(
        "--spectra",
        type=str,
        required=True,
        help="Path to spectra.csv matrix file.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels.csv file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save visualization figures.",
    )
    args = parser.parse_args()

    spectra_path = Path(args.spectra)
    labels_path = Path(args.labels)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelengths, reflectance_df = load_spectra_matrix(spectra_path)
    labels = load_labels(labels_path, list(reflectance_df.columns))

    class_means = compute_class_means(wavelengths, reflectance_df, labels)
    examples = select_example_spectra(wavelengths, reflectance_df, labels)

    mean_fig_path = plot_class_means(class_means, output_dir)
    examples_fig_path = plot_example_spectra(examples, output_dir)

    print("Visualization figures saved:")
    print(f"  {mean_fig_path}")
    print(f"  {examples_fig_path}")


if __name__ == "__main__":
    main()

