import argparse
import os
import sys
from typing import Dict

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluation.core import (
    build_dataset,
    compute_binary_metrics,
    ensure_dir,
    load_experiment_data,
    plot_metric_bars,
    predict_binary_deep_model,
    save_json,
    set_global_seed,
    train_binary_deep_model,
)
from evaluation.models import FusionAblationNet, get_ablation_builders


def _plot_ablation_tradeoff(summary_df: pd.DataFrame, output_dir: str, ablation_type: str) -> None:
    if summary_df.empty:
        return
    part = summary_df[summary_df["AblationType"] == ablation_type].copy()
    if part.empty:
        return
    x = part["PR_AUC_mean"].values
    y = part["F1_mean"].values
    mcc = part["MCC_mean"].fillna(0.0).values
    size = np.clip(mcc, 0.0, 1.0) * 900.0 + 120.0
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=size, alpha=0.75, c="#F58518", edgecolors="black", linewidth=0.6)
    for _, row in part.iterrows():
        plt.text(float(row["PR_AUC_mean"]) + 0.002, float(row["F1_mean"]) + 0.002, str(row["Model"]), fontsize=8)
    plt.xlabel("PR_AUC (mean)")
    plt.ylabel("F1 (mean)")
    plt.title(f"{ablation_type} Trade-off (bubble size = MCC)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ablation_{ablation_type}_tradeoff_scatter.png"), dpi=300)
    plt.close()


def _train_eval_setting(
    model_name: str,
    model,
    exp,
    device: torch.device,
    loss_cfg: Dict[str, float],
) -> Dict[str, float]:
    train_ds = build_dataset(exp, exp.train_samples, augmentation=True)
    val_ds = build_dataset(exp, exp.val_samples, augmentation=False)
    test_ds = build_dataset(exp, exp.test_samples, augmentation=False)

    train_labels = np.array([s[2] for s in exp.train_samples], dtype=np.int64)
    bg_count = max(1.0, float((train_labels == 0).sum()))
    pos_count = max(1.0, float((train_labels == 1).sum()))
    pos_weight = bg_count / pos_count

    trained = train_binary_deep_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        num_epochs=int(exp.config.get("training.num_epochs", 20)),
        batch_size=int(exp.config.get("training.batch_size", 8)),
        num_workers=int(exp.config.get("training.num_workers", 0)),
        learning_rate=float(exp.config.get("training.learning_rate", 0.001)),
        weight_decay=float(exp.config.get("training.weight_decay", 1e-4)),
        loss_cfg=loss_cfg,
        pos_weight=pos_weight,
        early_stopping_patience=int(exp.config.get("training.early_stopping_patience", 8)),
        verbose=True,
        log_prefix=f"[Ablation][{model_name}]",
    )
    pred = predict_binary_deep_model(
        model=trained["model"],
        dataset=test_ds,
        device=device,
        batch_size=int(exp.config.get("training.batch_size", 8)),
        num_workers=int(exp.config.get("training.num_workers", 0)),
    )
    metrics = compute_binary_metrics(y_true=pred["y_true"], y_prob=pred["y_prob"], threshold=0.5)
    metrics["Model"] = model_name
    return metrics


def _resolve_scheme(ablation_type: str, scheme: str) -> str:
    options = {
        "Architecture": ["Full", "NoPhysics", "SmallOnly", "LargeOnly", "PhysicsConcat", "PhysicsDiffNoAtt", "FCHead", "LightHead"],
        "Loss": ["WeightedBCE", "WeightedBCE_HNM", "WeightedBCE_Margin", "FullLoss", "StandardBCE"],
        "Window": ["Window_15", "Window_25", "Window_31"],
        "HNM_Ratio": ["HNM_0.25", "HNM_0.50", "HNM_0.75", "HNM_1.00"],
        "Margin": ["Margin_0.04", "Margin_0.06", "Margin_0.08"],
    }
    defaults = {
        "Architecture": "Full",
        "Loss": "FullLoss",
        "Window": "Window_25",
        "HNM_Ratio": "HNM_0.50",
        "Margin": "Margin_0.06",
    }
    if scheme == "auto":
        return defaults[ablation_type]
    if scheme not in options[ablation_type]:
        raise ValueError(f"AblationType={ablation_type} 不支持方案 {scheme}。可选：{options[ablation_type]}")
    return scheme


def _refresh_reports(output_dir: str, detail_df: pd.DataFrame) -> None:
    if detail_df.empty:
        return

    detail_df = detail_df.sort_values(["AblationType", "Model", "Seed"]).reset_index(drop=True)
    detail_csv = os.path.join(output_dir, "ablation_metrics_detail.csv")
    summary_csv = os.path.join(output_dir, "ablation_metrics_summary.csv")
    leaderboard_csv = os.path.join(output_dir, "ablation_leaderboard.csv")

    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    summary_df = detail_df.groupby(["AblationType", "Model"]).agg(
        {
            "Precision": ["mean", "std"],
            "Recall": ["mean", "std"],
            "F1": ["mean", "std"],
            "MCC": ["mean", "std"],
            "PR_AUC": ["mean", "std"],
            "ROC_AUC": ["mean", "std"],
            "Brier": ["mean", "std"],
        }
    )
    summary_df.columns = [f"{a}_{b}" for a, b in summary_df.columns]
    summary_df = summary_df.reset_index()
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    leaderboard = summary_df[["AblationType", "Model", "F1_mean", "PR_AUC_mean", "MCC_mean"]].copy()
    leaderboard["Rank_F1"] = leaderboard.groupby("AblationType")["F1_mean"].rank(ascending=False, method="min")
    leaderboard["Rank_PR_AUC"] = leaderboard.groupby("AblationType")["PR_AUC_mean"].rank(ascending=False, method="min")
    leaderboard["Rank_MCC"] = leaderboard.groupby("AblationType")["MCC_mean"].rank(ascending=False, method="min")
    leaderboard["Rank_Sum"] = leaderboard["Rank_F1"] + leaderboard["Rank_PR_AUC"] + leaderboard["Rank_MCC"]
    leaderboard = leaderboard.sort_values(["AblationType", "Rank_Sum", "F1_mean"], ascending=[True, True, False])
    leaderboard.to_csv(leaderboard_csv, index=False, encoding="utf-8-sig")

    for ab_type in summary_df["AblationType"].unique():
        part = summary_df[summary_df["AblationType"] == ab_type].copy()
        part_plot = pd.DataFrame(
            {
                "Model": part["Model"],
                "F1": part["F1_mean"],
                "PR_AUC": part["PR_AUC_mean"],
                "MCC": part["MCC_mean"],
            }
        )
        plot_metric_bars(
            metric_table=part_plot,
            metrics=["F1", "PR_AUC", "MCC"],
            out_dir=output_dir,
            prefix=f"ablation_{ab_type}",
        )
        _plot_ablation_tradeoff(summary_df=summary_df, output_dir=output_dir, ablation_type=ab_type)

    save_json(
        {
            "detail_csv": "ablation_metrics_detail.csv",
            "summary_csv": "ablation_metrics_summary.csv",
            "leaderboard_csv": "ablation_leaderboard.csv",
            "ablation_types": sorted(summary_df["AblationType"].unique().tolist()),
            "note": "消融脚本改为单方案增量执行。每次完成一个方案后立即更新汇总和图表。",
        },
        os.path.join(output_dir, "ablation_report_index.json"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 AG-S2CNN 消融实验（单方案增量执行版）")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/evaluation_outputs/ablation",
    )
    parser.add_argument(
        "--ablation-type",
        type=str,
        choices=["Architecture", "Loss", "Window", "HNM_Ratio", "Margin"],
        default="Architecture",
    )
    parser.add_argument("--scheme", type=str, default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    device = torch.device("cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    builders = get_ablation_builders()

    default_loss_cfg = {
        "hnm_ratio": 0.5,
        "hnm_weight": 0.2,
        "separation_weight": 0.2,
        "separation_margin": 0.06,
    }
    loss_variants = {
        "StandardBCE": {"hnm_ratio": 0.5, "hnm_weight": 0.0, "separation_weight": 0.0, "separation_margin": 0.06},
        "WeightedBCE": {"hnm_ratio": 0.5, "hnm_weight": 0.0, "separation_weight": 0.0, "separation_margin": 0.06},
        "WeightedBCE_HNM": {"hnm_ratio": 0.5, "hnm_weight": 0.2, "separation_weight": 0.0, "separation_margin": 0.06},
        "WeightedBCE_Margin": {"hnm_ratio": 0.5, "hnm_weight": 0.0, "separation_weight": 0.2, "separation_margin": 0.06},
        "FullLoss": {"hnm_ratio": 0.5, "hnm_weight": 0.2, "separation_weight": 0.2, "separation_margin": 0.06},
    }

    selected_scheme = _resolve_scheme(args.ablation_type, args.scheme)
    detail_csv = os.path.join(args.output_dir, "ablation_metrics_detail.csv")
    if os.path.exists(detail_csv):
        detail_df = pd.read_csv(detail_csv, encoding="utf-8-sig")
    else:
        detail_df = pd.DataFrame()

    print(f"[开始] 消融实验 type={args.ablation_type}, scheme={selected_scheme}, seeds={args.seeds}")
    for idx, seed in enumerate(args.seeds, start=1):
        print(f"[进度] seed {idx}/{len(args.seeds)} -> {seed}")
        set_global_seed(seed)
        exp = load_experiment_data(config_path=args.config, seed=seed, train_ratio=0.6, val_ratio=0.2)
        bands = exp.image_cube.shape[2]
        spatial = int(exp.config.get("model.spatial_size", 25))

        if args.ablation_type == "Architecture":
            model = builders[selected_scheme](bands, spatial)
            metrics = _train_eval_setting(
                model_name=selected_scheme,
                model=model,
                exp=exp,
                device=device,
                loss_cfg=default_loss_cfg,
            )
        elif args.ablation_type == "Loss":
            model = builders["Full"](bands, spatial)
            metrics = _train_eval_setting(
                model_name=selected_scheme,
                model=model,
                exp=exp,
                device=device,
                loss_cfg=loss_variants[selected_scheme],
            )
        elif args.ablation_type == "Window":
            window_size = int(selected_scheme.split("_")[1])
            exp_win = load_experiment_data(
                config_path=args.config,
                seed=seed,
                train_ratio=0.6,
                val_ratio=0.2,
                spatial_size_override=window_size,
            )
            model = FusionAblationNet(
                num_bands=exp_win.image_cube.shape[2],
                spatial_size=window_size,
                backbone_mode="full",
                fusion_mode="full",
                head_mode="lightweight",
            )
            metrics = _train_eval_setting(
                model_name=selected_scheme,
                model=model,
                exp=exp_win,
                device=device,
                loss_cfg=default_loss_cfg,
            )
        elif args.ablation_type == "HNM_Ratio":
            ratio = float(selected_scheme.split("_")[1])
            cfg = dict(default_loss_cfg)
            cfg["hnm_ratio"] = ratio
            model = builders["Full"](bands, spatial)
            metrics = _train_eval_setting(
                model_name=selected_scheme,
                model=model,
                exp=exp,
                device=device,
                loss_cfg=cfg,
            )
        else:
            margin = float(selected_scheme.split("_")[1])
            cfg = dict(default_loss_cfg)
            cfg["separation_margin"] = margin
            model = builders["Full"](bands, spatial)
            metrics = _train_eval_setting(
                model_name=selected_scheme,
                model=model,
                exp=exp,
                device=device,
                loss_cfg=cfg,
            )

        metrics["Seed"] = seed
        metrics["AblationType"] = args.ablation_type
        detail_df = pd.concat([detail_df, pd.DataFrame([metrics])], ignore_index=True)
        _refresh_reports(output_dir=args.output_dir, detail_df=detail_df)
        print(
            f"[结果] seed={seed}, type={args.ablation_type}, scheme={selected_scheme}, "
            f"F1={metrics['F1']:.4f}, PR_AUC={metrics['PR_AUC']:.4f}, MCC={metrics['MCC']:.4f}"
        )

    print(f"[完成] 消融实验输出目录: {os.path.abspath(args.output_dir)}")
    print("[提示] 下一次运行请切换 --scheme，以增量方式继续补齐其它消融方案。")


if __name__ == "__main__":
    main()
