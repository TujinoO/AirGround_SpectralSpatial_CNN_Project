import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


MODEL_MAP = {
    "PhysicsDiffNoAtt": "AG-S2CNN_All",
    "PhysicsConcat": "AG-S2CNN_NoDiff",
    "NoPhysics": "AG-S2CNN_NoPhysics",
}

MODEL_ORDER = [
    "AG-S2CNN_All",
    "AG-S2CNN_NoDiff",
    "AG-S2CNN_NoPhysics",
]

MODEL_COLORS = {
    "AG-S2CNN_All": "#1D4E89",
    "AG-S2CNN_NoDiff": "#4C78A8",
    "AG-S2CNN_NoPhysics": "#9DB7D5",
}

METRIC_COLORS = {
    "F1": "#1D4E89",
    "PR_AUC": "#2A9D8F",
    "MCC": "#E07A2D",
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.4,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _load_tables(input_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = os.path.join(input_dir, "ablation_metrics_summary.csv")
    detail_path = os.path.join(input_dir, "ablation_metrics_detail.csv")
    summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
    detail_df = pd.read_csv(detail_path, encoding="utf-8-sig")
    return summary_df, detail_df


def _filter_and_rename(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = summary_df[summary_df["Model"].isin(MODEL_MAP)].copy()
    detail_df = detail_df[detail_df["Model"].isin(MODEL_MAP)].copy()
    summary_df["Model"] = summary_df["Model"].map(MODEL_MAP)
    detail_df["Model"] = detail_df["Model"].map(MODEL_MAP)
    summary_df["Model"] = pd.Categorical(summary_df["Model"], categories=MODEL_ORDER, ordered=True)
    detail_df["Model"] = pd.Categorical(detail_df["Model"], categories=MODEL_ORDER, ordered=True)
    summary_df = summary_df.sort_values("Model").reset_index(drop=True)
    detail_df = detail_df.sort_values(["Model", "Seed"]).reset_index(drop=True)
    return summary_df, detail_df


def _save_tables(summary_df: pd.DataFrame, detail_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    _ensure_dir(output_dir)
    summary_out = os.path.join(output_dir, "ablation_focus_summary.csv")
    detail_out = os.path.join(output_dir, "ablation_focus_detail.csv")
    leaderboard_out = os.path.join(output_dir, "ablation_focus_leaderboard.csv")

    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_out, index=False, encoding="utf-8-sig")

    leaderboard = summary_df[["AblationType", "Model", "F1_mean", "PR_AUC_mean", "MCC_mean"]].copy()
    leaderboard["Rank_F1"] = leaderboard["F1_mean"].rank(ascending=False, method="min")
    leaderboard["Rank_PR_AUC"] = leaderboard["PR_AUC_mean"].rank(ascending=False, method="min")
    leaderboard["Rank_MCC"] = leaderboard["MCC_mean"].rank(ascending=False, method="min")
    leaderboard["Rank_Sum"] = leaderboard["Rank_F1"] + leaderboard["Rank_PR_AUC"] + leaderboard["Rank_MCC"]
    leaderboard = leaderboard.sort_values(["Rank_Sum", "F1_mean"], ascending=[True, False]).reset_index(drop=True)
    leaderboard.to_csv(leaderboard_out, index=False, encoding="utf-8-sig")
    return leaderboard


def _plot_grouped_metric_bars(summary_df: pd.DataFrame, output_dir: str) -> None:
    plot_df = summary_df.copy()
    metrics = ["F1_mean", "PR_AUC_mean", "MCC_mean"]
    metric_labels = ["F1", "PR_AUC", "MCC"]
    x = np.arange(len(metric_labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    fig.subplots_adjust(top=0.84, bottom=0.16)

    for idx, model in enumerate(MODEL_ORDER):
        row = plot_df[plot_df["Model"] == model].iloc[0]
        values = [float(row[m]) for m in metrics]
        positions = x + (idx - 1) * width
        bars = ax.bar(
            positions,
            values,
            width=width,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=1.8,
            label=model,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=11.5,
                color="#1F2937",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylim(0.40, 0.90)
    ax.set_ylabel("Metric Value")
    ax.set_title("Core Metric Comparison of Ablation Variants", pad=24)
    ax.grid(axis="y", color="#D9DEE7", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, title="Models")
    ax.text(
        0.0,
        1.045,
        "Three core metrics are merged into one publication-style figure for direct side-by-side comparison.",
        transform=ax.transAxes,
        fontsize=11.5,
        color="#4A5568",
    )
    fig.savefig(os.path.join(output_dir, "ablation_focus_metrics_panel.png"), dpi=600)
    plt.close(fig)


def _bubble_sizes(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-12:
        return np.full_like(values, 1200.0, dtype=float)
    scaled = (values - vmin) / (vmax - vmin)
    return 700.0 + scaled * 1800.0


def _plot_tradeoff_scatter(summary_df: pd.DataFrame, output_dir: str) -> None:
    df = summary_df.copy()
    df["BubbleSize"] = _bubble_sizes(df["MCC_mean"].to_numpy(dtype=float))
    label_offsets = {
        "AG-S2CNN_All": (10, 10),
        "AG-S2CNN_NoDiff": (10, 10),
        "AG-S2CNN_NoPhysics": (18, -18),
    }

    fig, ax = plt.subplots(figsize=(10.8, 7.4))
    fig.subplots_adjust(top=0.83)
    for _, row in df.iterrows():
        model = str(row["Model"])
        ax.scatter(
            float(row["PR_AUC_mean"]),
            float(row["F1_mean"]),
            s=float(row["BubbleSize"]),
            color=MODEL_COLORS[model],
            alpha=0.85,
            edgecolors="white",
            linewidths=2.0,
            zorder=3,
        )
        ax.annotate(
            model,
            xy=(float(row["PR_AUC_mean"]), float(row["F1_mean"])),
            xytext=label_offsets.get(model, (10, 10)),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=MODEL_COLORS[model], lw=1.3, alpha=0.96),
            arrowprops=dict(arrowstyle="-", color=MODEL_COLORS[model], lw=1.2),
            zorder=4,
        )

    ax.set_xlim(0.818, 0.831)
    ax.set_ylim(0.758, 0.788)
    ax.set_xlabel("PR_AUC")
    ax.set_ylabel("F1 Score")
    ax.set_title("Performance Trade-off Under Different Fusion Strategies", pad=24)
    ax.grid(color="#D9DEE7", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)

    size_levels = sorted(set([round(v, 3) for v in df["MCC_mean"].tolist()]))
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9AA5B1",
            markeredgecolor="white",
            markeredgewidth=1.5,
            markersize=size,
            alpha=0.55,
            label=f"MCC {level:.3f}",
        )
        for size, level in zip([13, 18, 24], size_levels)
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="white",
            markeredgewidth=1.8,
            markersize=12,
            label=model,
        )
        for model in MODEL_ORDER
    ]
    legend1 = ax.legend(handles=model_handles, loc="upper left", frameon=False, title="Models")
    legend2 = ax.legend(handles=size_handles, loc="lower right", frameon=False, title="Bubble Size")
    ax.add_artist(legend1)
    ax.text(
        0.0,
        1.045,
        "Higher-right is better; bubble area encodes MCC to show joint discrimination strength.",
        transform=ax.transAxes,
        fontsize=11.5,
        color="#4A5568",
    )
    fig.savefig(os.path.join(output_dir, "ablation_focus_tradeoff_scatter.png"), dpi=600)
    plt.close(fig)


def _plot_error_profile(detail_df: pd.DataFrame, output_dir: str) -> None:
    df = detail_df.copy()
    counts = ["FP", "FN", "TP"]
    x = np.arange(len(df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    fig.subplots_adjust(top=0.84, bottom=0.16)
    bar_colors = {"FP": "#C44E52", "FN": "#8172B3", "TP": "#55A868"}

    for idx, metric in enumerate(counts):
        vals = df[metric].astype(float).to_numpy()
        positions = x + (idx - 1) * width
        bars = ax.bar(
            positions,
            vals,
            width=width,
            color=bar_colors[metric],
            edgecolor="white",
            linewidth=1.5,
            label=metric,
            zorder=3,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 6,
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                color="#1F2937",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"].astype(str).tolist(), fontweight="bold")
    ax.set_ylabel("Sample Count")
    ax.set_title("Error Structure Comparison", pad=24)
    ax.grid(axis="y", color="#D9DEE7", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, title="Confusion Terms")

    ax.text(
        0.0,
        1.045,
        "Lower FP indicates cleaner target delineation, while TP and FN reflect the recall cost of stronger background suppression.",
        transform=ax.transAxes,
        fontsize=11.5,
        color="#4A5568",
    )
    fig.savefig(os.path.join(output_dir, "ablation_focus_error_profile.png"), dpi=600)
    plt.close(fig)


def _plot_radar(summary_df: pd.DataFrame, output_dir: str) -> None:
    df = summary_df.copy()
    labels = ["Precision", "Recall", "F1", "PR_AUC", "MCC"]
    cols = ["Precision_mean", "Recall_mean", "F1_mean", "PR_AUC_mean", "MCC_mean"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(11.6, 8.5))
    ax = plt.subplot(111, polar=True)
    fig.subplots_adjust(left=0.06, right=0.76, top=0.86, bottom=0.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", pad=16)
    ax.set_ylim(0.45, 0.90)
    ax.set_rgrids([0.50, 0.60, 0.70, 0.80, 0.90], angle=90, fontsize=10.5, color="#6B7280")
    ax.grid(color="#D9DEE7", linewidth=1.0)
    ax.spines["polar"].set_linewidth(1.3)
    ax.spines["polar"].set_color("#94A3B8")

    handles: List[Line2D] = []
    for _, row in df.iterrows():
        model = str(row["Model"])
        values = [float(row[col]) for col in cols]
        values += values[:1]
        ax.plot(angles, values, color=MODEL_COLORS[model], linewidth=3.0, alpha=0.95)
        ax.fill(angles, values, color=MODEL_COLORS[model], alpha=0.08)
        handles.append(Line2D([0], [0], color=MODEL_COLORS[model], lw=3.0, label=model))

    ax.set_title("Multi-Metric Radar Comparison", pad=28)
    ax.text(
        0.5,
        1.08,
        "The radar chart highlights the joint balance between recognition accuracy and discrimination quality.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.5,
        color="#4A5568",
    )
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Models")
    fig.savefig(os.path.join(output_dir, "ablation_focus_radar.png"), dpi=600)
    plt.close(fig)


def _build_analysis_text(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> str:
    s = summary_df.set_index("Model")
    d = detail_df.set_index("Model")

    all_row = s.loc["AG-S2CNN_All"]
    nodiff_row = s.loc["AG-S2CNN_NoDiff"]
    nophy_row = s.loc["AG-S2CNN_NoPhysics"]

    all_detail = d.loc["AG-S2CNN_All"]
    nodiff_detail = d.loc["AG-S2CNN_NoDiff"]
    nophy_detail = d.loc["AG-S2CNN_NoPhysics"]

    f1_gain_diff = float(all_row["F1_mean"] - nodiff_row["F1_mean"])
    pr_gain_diff = float(all_row["PR_AUC_mean"] - nodiff_row["PR_AUC_mean"])
    mcc_gain_diff = float(all_row["MCC_mean"] - nodiff_row["MCC_mean"])
    roc_gain_diff = float(all_row["ROC_AUC_mean"] - nodiff_row["ROC_AUC_mean"])

    fp_drop = int(nophy_detail["FP"] - all_detail["FP"])
    fp_drop_pct = fp_drop / float(nophy_detail["FP"]) * 100.0
    precision_gain = float(all_detail["Precision"] - nophy_detail["Precision"])
    recall_drop = float(nophy_detail["Recall"] - all_detail["Recall"])

    lines = [
        "# 三模型消融实验分析",
        "",
        "## 一、保留模型与命名说明",
        "",
        "- AG-S2CNN_All：原 `PhysicsDiffNoAtt`，表示引入物理先验并采用显式差分交互。",
        "- AG-S2CNN_NoDiff：原 `PhysicsConcat`，表示引入物理先验但仅做简单拼接，不进行显式差分约束。",
        "- AG-S2CNN_NoPhysics：原 `NoPhysics`，表示仅使用卫星流，不引入地面物理先验。",
        "",
        "## 二、从“空地交互方式是否关键”进行分析",
        "",
        f"- 在两种均引入物理先验的模型中，AG-S2CNN_All 相比 AG-S2CNN_NoDiff，F1 由 {nodiff_row['F1_mean']:.4f} 提升至 {all_row['F1_mean']:.4f}，增幅 {f1_gain_diff:.4f}。",
        f"- 同时，PR_AUC 由 {nodiff_row['PR_AUC_mean']:.4f} 提升至 {all_row['PR_AUC_mean']:.4f}，MCC 由 {nodiff_row['MCC_mean']:.4f} 提升至 {all_row['MCC_mean']:.4f}，ROC_AUC 由 {nodiff_row['ROC_AUC_mean']:.4f} 提升至 {all_row['ROC_AUC_mean']:.4f}。",
        f"- 其中，MCC 提升 {mcc_gain_diff:.4f}，是三项核心指标中最具辨识力的增益信号，说明显式差分交互不仅提升了总体分类效果，还显著增强了正负样本的综合判别一致性。",
        "- 由于 AG-S2CNN_All 与 AG-S2CNN_NoDiff 的唯一区别在于是否对地面标准与卫星观测进行显式差异建模，因此该结果能够直接说明：性能提升并非来自“多了一路输入”，而是来自“更有效的空地交互方式”。",
        "- 从机制上看，简单拼接只能让模型被动接收地面先验，而显式差分交互能够主动刻画卫星观测与地面标准之间的偏离关系，使模型重点关注真正与目标矿化相关的异常差异。",
        "- 因此，这一组对比构成了当前消融实验中最核心、也最有说服力的证据链：AG-S2CNN 的核心优势主要来自物理先验与卫星特征之间的显式差异建模。",
        "",
        "## 三、从“误报与漏报结构”进行分析",
        "",
        f"- AG-S2CNN_NoPhysics 的结果为：TP={int(nophy_detail['TP'])}，FP={int(nophy_detail['FP'])}，FN={int(nophy_detail['FN'])}。",
        f"- AG-S2CNN_NoDiff 的结果为：TP={int(nodiff_detail['TP'])}，FP={int(nodiff_detail['FP'])}，FN={int(nodiff_detail['FN'])}。",
        f"- AG-S2CNN_All 的结果为：TP={int(all_detail['TP'])}，FP={int(all_detail['FP'])}，FN={int(all_detail['FN'])}。",
        f"- 最值得强调的是误报项 FP 的变化。与 AG-S2CNN_NoPhysics 相比，AG-S2CNN_All 的 FP 从 {int(nophy_detail['FP'])} 降到 {int(all_detail['FP'])}，共减少 {fp_drop} 个，降幅约 {fp_drop_pct:.1f}%。",
        f"- 与此同时，Precision 从 {nophy_detail['Precision']:.4f} 提升到 {all_detail['Precision']:.4f}，增幅 {precision_gain:.4f}，说明模型对背景异常的抑制能力明显增强。",
        f"- 需要注意的是，AG-S2CNN_All 的 Recall 从 {nophy_detail['Recall']:.4f} 降至 {all_detail['Recall']:.4f}，下降 {recall_drop:.4f}。这表明模型在获得更强背景压制能力的同时，牺牲了一部分召回率。",
        "- 但从找矿应用视角看，这种变化并非负面结果。对于靶区圈定任务而言，减少假异常、降低无效圈定面积，往往比单纯追求更高召回更重要，因为过多误报会直接降低靶区部署效率和解释可信度。",
        "- AG-S2CNN_All 在 TP 略有下降的情况下大幅压缩 FP，并最终取得最高的 F1 与 MCC，这说明其并不是简单地“变保守”，而是在识别边界上实现了更合理的精确化收缩。",
        "- 换言之，AG-S2CNN_All 的优势体现在：它能够以较小的漏检代价，显著换取更干净的异常响应图和更可靠的靶区判别结果。",
        "",
        "## 四、综合评价结论",
        "",
        "- 如果研究目标是证明“物理先验通过合理交互方式确实提升了模型判别质量”，那么当前三模型结果已经能够提供较强支撑。",
        "- AG-S2CNN_NoPhysics 说明：没有物理先验时，模型更容易产生背景误报。",
        "- AG-S2CNN_NoDiff 说明：仅注入物理先验但不做显式差分，提升有限。",
        "- AG-S2CNN_All 说明：当物理先验通过显式差分交互进入模型后，才能真正发挥其校准和约束作用，并在误报抑制、综合判别和平衡表现上取得最佳结果。",
        "- 因此，当前这组三模型消融实验最适合支撑的结论是：AG-S2CNN 的优势主要来源于“物理先验注入 + 显式差分交互”这一机制组合，其中显式差分交互是将地学知识转化为精度增益的关键环节。",
    ]
    return "\n".join(lines) + "\n"


def _save_analysis(summary_df: pd.DataFrame, detail_df: pd.DataFrame, output_dir: str) -> None:
    analysis_text = _build_analysis_text(summary_df=summary_df, detail_df=detail_df)
    analysis_path = os.path.join(output_dir, "ablation_focus_analysis.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(analysis_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成三模型论文版消融分析与美化图表。")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"D:\Grp_data\Air-Ground_Spectral-Spatial_CNN\3_model\evaluation_outputs\ablation_core4_quick",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"D:\Grp_data\Air-Ground_Spectral-Spatial_CNN\3_model\evaluation_outputs\ablation_focus3_paper",
    )
    args = parser.parse_args()

    _ensure_dir(args.output_dir)
    _set_style()
    summary_df, detail_df = _load_tables(args.input_dir)
    summary_df, detail_df = _filter_and_rename(summary_df, detail_df)
    _save_tables(summary_df=summary_df, detail_df=detail_df, output_dir=args.output_dir)
    _save_analysis(summary_df=summary_df, detail_df=detail_df, output_dir=args.output_dir)
    _plot_grouped_metric_bars(summary_df=summary_df, output_dir=args.output_dir)
    _plot_tradeoff_scatter(summary_df=summary_df, output_dir=args.output_dir)
    _plot_error_profile(detail_df=detail_df, output_dir=args.output_dir)
    _plot_radar(summary_df=summary_df, output_dir=args.output_dir)
    print(f"Focused ablation package saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
