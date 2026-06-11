#!/usr/bin/env python3
"""
Redraw six output visualizations with unified paper style using real data.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter


CLASS_NAMES_CN = {
    0: "锂辉石型富矿伟晶岩",
    1: "锂云母型富矿伟晶岩",
    2: "混合型富矿伟晶岩",
    3: "贫矿伟晶岩",
    4: "围岩",
}

PALETTE = ["#D49CA6", "#8FA8C7", "#9CC5AF", "#E3B793", "#B9A5D4"]
SAMPLE_PALETTE = ["#E7B7B2", "#F1CF96", "#C7DCA0", "#9FCBBC", "#9FB3D7"]
MEAN_CURVE_COLOR = "#C73A32"
GSRSL_HIGHLIGHT_PALETTE = ["#D7263D", "#1D4ED8", "#15803D", "#D97706", "#6D28D9"]


def parse_metadata(metadata_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    text = metadata_path.read_text(encoding="utf-8", errors="ignore")
    w_pattern = re.compile(r"Wavelengths\s+(\d+)\s*=\s*([0-9.]+)")
    f_pattern = re.compile(r"FWHM\s+(\d+)\s*=\s*([0-9.]+)")

    w_map: Dict[int, float] = {int(i): float(v) for i, v in w_pattern.findall(text)}
    f_map: Dict[int, float] = {int(i): float(v) for i, v in f_pattern.findall(text)}
    idx = sorted(set(w_map.keys()) & set(f_map.keys()))
    if len(idx) != 297:
        raise ValueError(f"metadata bands != 297, got {len(idx)}")

    wavelengths = np.array([w_map[i] for i in idx], dtype=np.float64)
    fwhms = np.array([f_map[i] for i in idx], dtype=np.float64)
    return wavelengths, fwhms


def choose_font() -> str:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def apply_style() -> None:
    selected_font = choose_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected_font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def gaussian_resample_one(
    wl_ground: np.ndarray,
    refl_ground: np.ndarray,
    sat_wl: np.ndarray,
    sat_fwhm: np.ndarray,
) -> np.ndarray:
    sigmas = sat_fwhm / 2.355
    out = np.full_like(sat_wl, np.nan, dtype=np.float64)
    for i, center in enumerate(sat_wl):
        sigma = sigmas[i]
        left, right = center - 3.0 * sigma, center + 3.0 * sigma
        mask = (wl_ground >= left) & (wl_ground <= right)
        if not np.any(mask):
            continue
        w = np.exp(-((wl_ground[mask] - center) ** 2) / (2.0 * sigma * sigma))
        out[i] = float(np.sum(refl_ground[mask] * w) / np.sum(w))
    return out


def select_representative_indices(
    spectra: List[np.ndarray],
    class_mean: np.ndarray,
    anchor: np.ndarray,
    n_samples: int = 5,
) -> List[int]:
    if len(spectra) <= n_samples:
        return list(range(len(spectra)))

    mean_scale = max(float(np.ptp(class_mean)), 1e-6)
    anchor_scale = max(float(np.ptp(anchor)), 1e-6)
    scores: List[Tuple[float, int]] = []
    for idx, spectrum in enumerate(spectra):
        mean_rmse = float(np.sqrt(np.mean((spectrum - class_mean) ** 2)) / mean_scale)
        anchor_rmse = float(np.sqrt(np.mean((spectrum - anchor) ** 2)) / anchor_scale)
        scores.append((0.55 * anchor_rmse + 0.45 * mean_rmse, idx))
    return [idx for _, idx in sorted(scores)[:n_samples]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root folder with metadata/labels/spectra/output")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = root / "GF5A_AHSI_20240126__metadata.txt"
    labels_path = root / "labels.csv"
    spectra_path = root / "spectra.csv"
    gsrsl_path = output_dir / "gsrsl.npy"

    sat_wl, sat_fwhm = parse_metadata(metadata_path)
    label_df = pd.read_csv(labels_path)
    label_map = dict(zip(label_df["filename"].astype(str), label_df["class_id"].astype(int)))

    spectra_df = pd.read_csv(spectra_path)
    wl_ground = spectra_df.iloc[:, 0].to_numpy(dtype=np.float64)
    spec_cols = [c for c in spectra_df.columns[1:] if str(c) in label_map]

    by_class_raw: Dict[int, List[np.ndarray]] = {i: [] for i in range(5)}
    by_class_smooth: Dict[int, List[np.ndarray]] = {i: [] for i in range(5)}
    by_class_files: Dict[int, List[str]] = {i: [] for i in range(5)}
    example_payload: Dict[int, Dict[str, np.ndarray | str]] = {}
    anomalies: Dict[int, int] = {i: 0 for i in range(5)}

    for col in spec_cols:
        class_id = int(label_map[str(col)])
        raw = spectra_df[col].to_numpy(dtype=np.float64)
        is_anomaly = bool(np.any(raw < 0.0) or np.any(raw > 1.0))
        if is_anomaly:
            anomalies[class_id] += 1
        raw = np.clip(raw, 0.0, 1.0)
        smooth = np.clip(savgol_filter(raw, window_length=15, polyorder=3, mode="interp"), 0.0, 1.0)
        by_class_raw[class_id].append(raw)
        by_class_smooth[class_id].append(smooth)
        by_class_files[class_id].append(str(col))
        if class_id not in example_payload:
            example_payload[class_id] = {
                "filename": str(col),
                "raw": raw,
                "smooth": smooth,
            }

    class_counts = np.array([len(by_class_raw[i]) for i in range(5)], dtype=np.int32)

    smoothed_class_means: Dict[int, np.ndarray] = {}
    for cid in range(5):
        mat = np.vstack(by_class_raw[cid])
        mean_raw = np.mean(mat, axis=0)
        smoothed_class_means[cid] = np.clip(
            savgol_filter(mean_raw, window_length=15, polyorder=3, mode="interp"), 0.0, 1.0
        )

    if gsrsl_path.exists():
        class_means_sat = np.load(gsrsl_path, allow_pickle=True).item()
    else:
        class_means_sat = {
            cid: gaussian_resample_one(wl_ground, smoothed_class_means[cid], sat_wl, sat_fwhm).astype(np.float32)
            for cid in range(5)
        }

    primary_class = 0
    first_examples = sorted(example_payload.keys())
    if len(first_examples) > 0:
        primary_class = first_examples[0]
    ex = example_payload[primary_class]
    ex_raw = np.asarray(ex["raw"], dtype=np.float64)
    ex_smooth = np.asarray(ex["smooth"], dtype=np.float64)
    ex_sat = gaussian_resample_one(wl_ground, ex_smooth, sat_wl, sat_fwhm)

    apply_style()

    # 1) gsrsl_visualization.png
    fig, ax = plt.subplots(figsize=(13, 7))
    for cid in range(5):
        ax.plot(
            sat_wl,
            class_means_sat[cid],
            color=GSRSL_HIGHLIGHT_PALETTE[cid],
            linewidth=4.0,
            alpha=0.98,
            label=f"类别{cid}：{CLASS_NAMES_CN[cid]}",
        )
    ax.set_xlabel("波长 / nm", fontsize=18)
    ax.set_ylabel("反射率", fontsize=18)
    ax.set_title("地面标准参考光谱库（GSRSL）", fontsize=21, pad=12)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(float(sat_wl[0]), float(sat_wl[-1]))
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(loc="upper right", frameon=True, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "gsrsl_visualization.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2) gsrsl_visualization_paper.png
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    for cid in range(5):
        ax.plot(sat_wl, class_means_sat[cid], color=PALETTE[cid], linewidth=2.9, label=f"类别{cid}")
    ax.set_xlabel("波长 / nm")
    ax.set_ylabel("反射率")
    ax.set_title("GSRSL论文展示图（统一低饱和配色）")
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.set_xlim(float(sat_wl[0]), float(sat_wl[-1]))
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=True, ncol=1, title="岩性类别")
    fig.tight_layout()
    fig.savefig(output_dir / "gsrsl_visualization_paper.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 3) real_spectra_class_means.png
    fig, ax = plt.subplots(figsize=(13, 7))
    for cid in range(5):
        ax.plot(wl_ground, smoothed_class_means[cid], color=PALETTE[cid], linewidth=2.7, label=f"类别{cid}：{CLASS_NAMES_CN[cid]}")
    ax.set_xlabel("波长 / nm")
    ax.set_ylabel("反射率")
    ax.set_title("各岩性类别地面光谱平滑均值对比")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(float(wl_ground[0]), float(wl_ground[-1]))
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "real_spectra_class_means.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 4) real_spectra_examples_smoothed.png
    available_classes = sorted(example_payload.keys())
    fig, axes = plt.subplots(len(available_classes), 1, figsize=(12.5, 2.65 * len(available_classes)), sharex=True)
    if len(available_classes) == 1:
        axes = [axes]
    panel_labels = list("abcdefghijklmnopqrstuvwxyz")
    for idx, cid in enumerate(available_classes):
        payload = example_payload[cid]
        raw = np.asarray(payload["raw"], dtype=np.float64)
        smooth = np.asarray(payload["smooth"], dtype=np.float64)
        ax = axes[idx]
        ax.plot(wl_ground, raw, color="#B8BEC7", linewidth=1.9, label="原始光谱")
        ax.plot(wl_ground, smooth, color=PALETTE[cid], linewidth=2.4, label="SG滤波后")
        ax.set_ylabel("反射率")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_title(f"类别{cid}：{CLASS_NAMES_CN[cid]} | 样本：{payload['filename']}")
        ax.text(0.01, 0.95, f"({panel_labels[idx]})", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
        ax.legend(loc="upper right", frameon=True)
    axes[-1].set_xlabel("波长 / nm")
    fig.tight_layout()
    fig.savefig(output_dir / "real_spectra_examples_smoothed.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 5) representative_spectra_with_class_means.png
    fig, axes = plt.subplots(5, 1, figsize=(16, 23), sharex=True)
    representative_handles = [
        Line2D([0], [0], color=SAMPLE_PALETTE[0], linewidth=1.7, label="5条典型样本"),
        Line2D([0], [0], color=MEAN_CURVE_COLOR, linewidth=3.8, label="该类平均光谱"),
    ]
    for cid, ax in enumerate(axes):
        class_mean = smoothed_class_means[cid]
        anchor = np.asarray(example_payload[cid]["smooth"], dtype=np.float64)
        selected_indices = select_representative_indices(by_class_smooth[cid], class_mean, anchor, n_samples=5)
        offsets = np.linspace(-0.028, 0.028, len(selected_indices))
        stacked_curves: List[np.ndarray] = [class_mean]
        for rank, (sample_idx, offset) in enumerate(zip(selected_indices, offsets), start=1):
            shifted_curve = by_class_smooth[cid][sample_idx] + offset
            stacked_curves.append(shifted_curve)
            ax.plot(
                wl_ground,
                shifted_curve,
                color=SAMPLE_PALETTE[(rank - 1) % len(SAMPLE_PALETTE)],
                linewidth=1.55,
                alpha=0.94,
            )
        ax.plot(wl_ground, class_mean, color=MEAN_CURVE_COLOR, linewidth=3.6, alpha=0.98)
        curve_stack = np.vstack(stacked_curves)
        y_min = max(0.0, float(np.min(curve_stack)) - 0.03)
        y_max = min(1.0, float(np.max(curve_stack)) + 0.03)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("反射率", fontsize=17)
        ax.set_title(f"类别{cid}：{CLASS_NAMES_CN[cid]}", fontsize=18, pad=12)
        ax.grid(True, alpha=0.22, linestyle="--")
        ax.legend(handles=representative_handles, loc="upper right", frameon=True, fontsize=15)
        ax.tick_params(axis="both", labelsize=15)
    axes[-1].set_xlabel("波长 / nm", fontsize=18)
    axes[-1].set_xlim(float(wl_ground[0]), float(wl_ground[-1]))
    fig.suptitle("各岩性类别典型样本光谱与类别平均光谱对比", fontsize=22, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.986))
    fig.savefig(
        output_dir / "representative_spectra_with_class_means.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    # 6) savgol_filter_demo.png
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8.4))
    ax1.plot(wl_ground, ex_raw, color="#B8BEC7", linewidth=2.0, label="原始光谱")
    ax1.plot(wl_ground, ex_smooth, color=PALETTE[primary_class], linewidth=2.6, label="SG滤波后")
    ax1.set_xlim(float(wl_ground[0]), float(wl_ground[-1]))
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("波长 / nm")
    ax1.set_ylabel("反射率")
    ax1.set_title("Savitzky-Golay滤波：全波段对比")
    ax1.text(0.01, 0.95, "(a)", transform=ax1.transAxes, va="top", ha="left", fontweight="bold")
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.legend(loc="upper right", frameon=True)

    mask = (wl_ground >= 2000.0) & (wl_ground <= 2400.0)
    ax2.plot(wl_ground[mask], ex_raw[mask], color="#B8BEC7", linewidth=2.0, label="原始光谱")
    ax2.plot(wl_ground[mask], ex_smooth[mask], color=PALETTE[primary_class], linewidth=2.6, label="滤波后")
    ax2.set_xlabel("波长 / nm")
    ax2.set_ylabel("反射率")
    ax2.set_title("2200 nm附近吸收特征放大")
    ax2.text(0.01, 0.95, "(b)", transform=ax2.transAxes, va="top", ha="left", fontweight="bold")
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "savgol_filter_demo.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 7) resampler_demo_output.png
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8.4))
    ax1.plot(wl_ground, ex_smooth, color=PALETTE[primary_class], linewidth=2.5, label="地面高分辨率光谱（SG后）")
    ax1.set_xlabel("波长 / nm")
    ax1.set_ylabel("反射率")
    ax1.set_title("地面高分辨率连续光谱")
    ax1.text(0.01, 0.95, "(a)", transform=ax1.transAxes, va="top", ha="left", fontweight="bold")
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.set_xlim(float(wl_ground[0]), float(wl_ground[-1]))
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(loc="upper right", frameon=True)

    ax2.plot(wl_ground, ex_smooth, color="#C5CBD3", linewidth=1.9, label="地面光谱（连续）")
    ax2.plot(sat_wl, ex_sat, "o-", color=PALETTE[primary_class], markersize=4.5, linewidth=2.2, label="GF-5重采样结果（离散）")
    ax2.set_xlabel("波长 / nm")
    ax2.set_ylabel("反射率")
    ax2.set_title("地面光谱与GF-5波段重采样对比")
    ax2.text(0.01, 0.95, "(b)", transform=ax2.transAxes, va="top", ha="left", fontweight="bold")
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.set_xlim(float(wl_ground[0]), float(wl_ground[-1]))
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "resampler_demo_output.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Done. Updated 7 figures in:", output_dir)
    print("Class counts:", class_counts.tolist())
    print("Anomalies:", [int(anomalies[i]) for i in range(5)])


if __name__ == "__main__":
    main()
