import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


OUTPUT_DIR_DEFAULT = r"D:\Grp_data\Air-Ground_Spectral-Spatial_CNN\3_model\evaluation_outputs\comparative"


SUITE_COLORS = {
    "deep": "#2F5DA9",
    "ml": "#2A9D8F",
    "physical": "#E07A2D",
}


MODEL_COLORS = {
    "AG-S2CNN": "#1D4E89",
    "3D-CNN-Standard": "#4C78A8",
    "RandomForest": "#2A9D8F",
    "MLP": "#59A14F",
    "LogisticRegression": "#76B7B2",
    "SVM-RBF": "#8CD17D",
    "SAM": "#F28E2B",
    "SID": "#E15759",
    "SAM-SID": "#B07AA1",
    "CEM": "#9C755F",
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _suite_from_model(model: str) -> str:
    if model in {"AG-S2CNN", "3D-CNN-Standard"}:
        return "deep"
    if model in {"RandomForest", "MLP", "LogisticRegression", "SVM-RBF", "XGBoost"}:
        return "ml"
    return "physical"


def _set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.4,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _load_tables(output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_path = os.path.join(output_dir, "final_main_results.csv")
    full_path = os.path.join(output_dir, "final_full_metrics.csv")
    main_df = pd.read_csv(main_path, encoding="utf-8-sig")
    full_df = pd.read_csv(full_path, encoding="utf-8-sig")

    if "Suite" not in main_df.columns:
        main_df["Suite"] = main_df["Model"].map(_suite_from_model)
    if "Suite" not in full_df.columns:
        full_df["Suite"] = full_df["Model"].map(_suite_from_model)

    return main_df, full_df


def _plot_brier_bar(main_df: pd.DataFrame, output_dir: str) -> None:
    df = main_df.copy().sort_values("Brier", ascending=True).reset_index(drop=True)
    colors = [SUITE_COLORS.get(suite, "#6C757D") for suite in df["Suite"]]
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    fig.subplots_adjust(top=0.86, bottom=0.20)
    bars = ax.barh(
        y,
        df["Brier"].values,
        color=colors,
        height=0.72,
        edgecolor="#1F1F1F",
        linewidth=1.4,
    )

    for idx, (bar, model, value) in enumerate(zip(bars, df["Model"], df["Brier"])):
        if model == "AG-S2CNN":
            bar.set_edgecolor("#C62828")
            bar.set_linewidth(2.6)
        ax.text(
            float(value) + 0.004,
            idx,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=13,
            weight="bold" if idx == 0 else "normal",
            color="#222222",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["Model"])
    ax.invert_yaxis()
    ax.set_xlabel("Brier Score")
    ax.set_title("Calibration Error Comparison", pad=28)
    ax.grid(axis="x", color="#D9DEE7", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlim(0.0, float(df["Brier"].max()) + 0.06)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=SUITE_COLORS["deep"], lw=9, label="Deep Models"),
        Line2D([0], [0], color=SUITE_COLORS["ml"], lw=9, label="Machine Learning"),
        Line2D([0], [0], color=SUITE_COLORS["physical"], lw=9, label="Physical Methods"),
        Line2D([0], [0], color="#C62828", lw=2.8, label="Highlighted: AG-S2CNN"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        frameon=False,
        title="Groups",
        ncol=2,
    )

    ax.text(
        0.0,
        1.045,
        "Lower is better. Values are unchanged from the completed comparative experiment.",
        transform=ax.transAxes,
        fontsize=12,
        color="#4A5568",
    )

    fig.savefig(os.path.join(output_dir, "final_brier_bar.png"), dpi=600)
    plt.close(fig)


def _bubble_sizes(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-12:
        return np.full_like(values, 900.0, dtype=float)
    scaled = (values - vmin) / (vmax - vmin)
    return 450.0 + scaled * 1950.0


def _plot_bubble_scatter(main_df: pd.DataFrame, output_dir: str) -> None:
    df = main_df.copy().sort_values(["PR_AUC", "F1"], ascending=[True, True]).reset_index(drop=True)
    df["Suite"] = df["Suite"].fillna(df["Model"].map(_suite_from_model))
    df["Size"] = _bubble_sizes(df["MCC"].values.astype(float))

    fig, ax = plt.subplots(figsize=(12.8, 8.0))
    fig.subplots_adjust(top=0.82, right=0.93, bottom=0.10)

    for suite in ["deep", "ml", "physical"]:
        sub = df[df["Suite"] == suite]
        if sub.empty:
            continue
        ax.scatter(
            sub["PR_AUC"],
            sub["F1"],
            s=sub["Size"],
            c=SUITE_COLORS[suite],
            alpha=0.82,
            edgecolors="white",
            linewidths=2.0,
            label={
                "deep": "Deep Models",
                "ml": "Machine Learning",
                "physical": "Physical Methods",
            }[suite],
            zorder=3,
        )

    ag_row = df[df["Model"] == "AG-S2CNN"]
    if not ag_row.empty:
        ax.scatter(
            ag_row["PR_AUC"],
            ag_row["F1"],
            s=ag_row["Size"] * 1.08,
            facecolors="none",
            edgecolors="#C58B00",
            linewidths=3.0,
            zorder=4,
        )

    crowded = df[(df["PR_AUC"] >= 0.75) & (df["F1"] >= 0.70)].sort_values("F1", ascending=False).reset_index(drop=True)
    crowded_models = set(crowded["Model"].tolist())
    crowded_positions: Dict[str, tuple[float, float]] = {
        "MLP": (0.998, 0.872),
        "AG-S2CNN": (0.956, 0.846),
        "RandomForest": (0.942, 0.818),
        "SVM-RBF": (0.972, 0.790),
        "3D-CNN-Standard": (0.934, 0.758),
        "LogisticRegression": (0.930, 0.732),
    }

    for _, row in crowded.iterrows():
        text_x, text_y = crowded_positions.get(row["Model"], (0.97, float(row["F1"]) + 0.02))
        ax.annotate(
            row["Model"],
            xy=(float(row["PR_AUC"]), float(row["F1"])),
            xytext=(text_x, text_y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=12.5,
            fontweight="bold" if row["Model"] == "AG-S2CNN" else "normal",
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=MODEL_COLORS.get(row["Model"], "#666666"), lw=1.2, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color=MODEL_COLORS.get(row["Model"], "#666666"), lw=1.6, shrinkA=4, shrinkB=6),
            zorder=5,
        )

    small_offsets: Dict[str, tuple[float, float]] = {
        "CEM": (0.002, 0.010),
        "SAM": (0.003, 0.012),
        "SAM-SID": (0.003, 0.014),
        "SID": (0.003, 0.013),
    }
    for _, row in df[~df["Model"].isin(crowded_models)].iterrows():
        dx, dy = small_offsets.get(row["Model"], (0.004, 0.012))
        ax.text(
            float(row["PR_AUC"]) + dx,
            float(row["F1"]) + dy,
            row["Model"],
            fontsize=12.5,
            weight="bold" if row["Model"] == "AG-S2CNN" else "normal",
            color="#1F2937",
            zorder=5,
        )

    ax.set_xlim(0.48, 1.08)
    ax.set_ylim(0.35, 0.95)
    ax.set_xlabel("PR AUC")
    ax.set_ylabel("F1 Score")
    ax.set_title("Performance Trade-off Across Models", pad=28)
    ax.grid(color="#D9DEE7", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)

    group_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SUITE_COLORS["deep"], markeredgecolor="white", markeredgewidth=1.8, markersize=12, label="Deep Models"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SUITE_COLORS["ml"], markeredgecolor="white", markeredgewidth=1.8, markersize=12, label="Machine Learning"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SUITE_COLORS["physical"], markeredgecolor="white", markeredgewidth=1.8, markersize=12, label="Physical Methods"),
    ]
    size_levels = np.percentile(df["MCC"].values, [25, 50, 75]).tolist()
    size_levels = sorted(set([round(v, 2) for v in size_levels]))
    size_marker_sizes = [12, 18, 24][: len(size_levels)]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9AA5B1",
            markeredgecolor="white",
            markeredgewidth=1.5,
            alpha=0.45,
            markersize=size_marker_sizes[idx],
            label=f"MCC {level:.2f}",
        )
        for idx, level in enumerate(size_levels)
    ]

    legend1 = ax.legend(handles=group_handles, loc="upper left", frameon=False, title="Groups", labelspacing=0.7, handletextpad=0.6)
    legend2 = ax.legend(
        handles=size_handles,
        loc="lower right",
        frameon=False,
        title="Bubble Size",
        labelspacing=1.0,
        handletextpad=0.9,
        borderaxespad=0.8,
    )
    ax.add_artist(legend1)

    ax.text(
        0.0,
        1.045,
        "Bubble area encodes MCC. Labels are repositioned to avoid overlap while preserving the original data.",
        transform=ax.transAxes,
        fontsize=12,
        color="#4A5568",
    )

    fig.savefig(os.path.join(output_dir, "final_bubble_scatter.png"), dpi=600)
    plt.close(fig)


def _plot_radar_chart(full_df: pd.DataFrame, output_dir: str) -> None:
    required = ["Model", "F1", "PR_AUC", "MCC", "Brier", "Recall"]
    df = full_df[required].copy()
    df["OneMinusBrier"] = 1.0 - df["Brier"].astype(float)
    df = df.sort_values(["F1", "PR_AUC"], ascending=[False, False]).reset_index(drop=True)

    labels = ["F1", "PR AUC", "MCC", "1 - Brier", "Recall"]
    plot_cols = ["F1", "PR_AUC", "MCC", "OneMinusBrier", "Recall"]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(12.5, 9.2))
    ax = plt.subplot(111, polar=True)
    fig.subplots_adjust(left=0.05, right=0.72, top=0.84, bottom=0.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", pad=18)
    ax.set_ylim(0.0, 1.0)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=90, fontsize=11, color="#6B7280")
    ax.grid(color="#D9DEE7", linewidth=1.0)
    ax.spines["polar"].set_linewidth(1.3)
    ax.spines["polar"].set_color("#94A3B8")

    legend_handles: List[Line2D] = []
    for _, row in df.iterrows():
        model = row["Model"]
        values = [float(row[col]) for col in plot_cols]
        values += values[:1]
        color = MODEL_COLORS.get(model, "#4C78A8")
        lw = 3.2 if model == "AG-S2CNN" else 2.1
        alpha = 0.98 if model == "AG-S2CNN" else 0.78
        ax.plot(angles, values, color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")
        if model == "AG-S2CNN":
            ax.fill(angles, values, color=color, alpha=0.10, zorder=2)
        legend_handles.append(Line2D([0], [0], color=color, lw=lw, label=model))

    ax.set_title("Multi-Metric Radar Comparison", pad=36)
    ax.text(
        0.5,
        1.085,
        "All axes use the original metric values. The Brier axis is shown as 1 - Brier so that higher is better.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="#4A5568",
    )

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        title="Models",
        ncol=1,
    )

    fig.savefig(os.path.join(output_dir, "final_radar_chart.png"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="美化对比实验最终图表，不改变任何数据值。")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()

    _ensure_dir(args.output_dir)
    _set_style()
    main_df, full_df = _load_tables(args.output_dir)
    _plot_brier_bar(main_df=main_df, output_dir=args.output_dir)
    _plot_bubble_scatter(main_df=main_df, output_dir=args.output_dir)
    _plot_radar_chart(full_df=full_df, output_dir=args.output_dir)
    print(f"Beautified figures saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
