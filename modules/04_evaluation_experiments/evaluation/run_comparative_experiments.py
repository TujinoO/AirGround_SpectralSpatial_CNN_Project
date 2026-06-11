import argparse
import os
import re
import sys
from glob import glob
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluation.core import (
    build_dataset,
    compute_binary_metrics,
    ensure_dir,
    extract_center_spectra,
    extract_spectral_and_local_stats,
    load_experiment_data,
    paired_significance,
    plot_metric_bars,
    plot_probability_hist,
    plot_roc_pr_curves,
    predict_binary_deep_model,
    save_json,
    set_global_seed,
    train_binary_deep_model,
)
from evaluation.models import get_comparative_deep_builders


def _best_threshold_by_f1(y_true: np.ndarray, score: np.ndarray) -> float:
    candidates = np.linspace(float(score.min()), float(score.max()), 121)
    best_thr = float(np.median(score))
    best_f1 = -1.0
    for thr in candidates:
        pred = (score <= thr).astype(np.int64)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _sam_angle(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    num = np.sum(x * s[None, :], axis=1)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(s)
    val = np.clip(num / (den + 1e-12), -1.0, 1.0)
    return np.arccos(val)


def _sid_distance(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    x_pos = np.clip(x, 1e-8, None)
    s_pos = np.clip(s, 1e-8, None)
    p = x_pos / np.sum(x_pos, axis=1, keepdims=True)
    q = s_pos[None, :] / np.sum(s_pos)
    sid_pq = np.sum(p * np.log((p + 1e-12) / (q + 1e-12)), axis=1)
    sid_qp = np.sum(q * np.log((q + 1e-12) / (p + 1e-12)), axis=1)
    return sid_pq + sid_qp


def _cem_response(x: np.ndarray, d: np.ndarray, bg_x: np.ndarray) -> np.ndarray:
    cov = np.cov(bg_x.T) + 1e-6 * np.eye(bg_x.shape[1], dtype=np.float32)
    inv = np.linalg.pinv(cov)
    w = inv @ d
    w = w / (d.T @ inv @ d + 1e-12)
    return x @ w


def _run_physical_methods(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    target_ref: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}

    def _norm_by_source(raw: np.ndarray, src_min: float, src_max: float, reverse: bool) -> np.ndarray:
        score = (raw - src_min) / (src_max - src_min + 1e-12)
        score = np.clip(score, 0.0, 1.0)
        if reverse:
            score = 1.0 - score
        return score.astype(np.float32)

    sam_val = _sam_angle(x_val, target_ref)
    sam_thr = _best_threshold_by_f1(y_val, sam_val)
    sam_test = _sam_angle(x_test, target_ref)
    sam_min, sam_max = float(sam_val.min()), float(sam_val.max())
    out["SAM"] = {
        "val": _norm_by_source(sam_val, sam_min, sam_max, reverse=True),
        "test": _norm_by_source(sam_test, sam_min, sam_max, reverse=True),
    }
    _ = sam_thr

    sid_val = _sid_distance(x_val, target_ref)
    sid_thr = _best_threshold_by_f1(y_val, sid_val)
    sid_test = _sid_distance(x_test, target_ref)
    sid_min, sid_max = float(sid_val.min()), float(sid_val.max())
    out["SID"] = {
        "val": _norm_by_source(sid_val, sid_min, sid_max, reverse=True),
        "test": _norm_by_source(sid_test, sid_min, sid_max, reverse=True),
    }
    _ = sid_thr

    best_alpha = 0.5
    best_f1 = -1.0
    for alpha in np.linspace(0.0, 1.0, 11):
        mix = alpha * sam_val + (1.0 - alpha) * sid_val
        thr = _best_threshold_by_f1(y_val, mix)
        f1 = f1_score(y_val, (mix <= thr).astype(np.int64), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = float(alpha)
    mix_test = best_alpha * _sam_angle(x_test, target_ref) + (1.0 - best_alpha) * _sid_distance(x_test, target_ref)
    mix_min, mix_max = float(mix.min()), float(mix.max())
    out["SAM-SID"] = {
        "val": _norm_by_source(mix, mix_min, mix_max, reverse=True),
        "test": _norm_by_source(mix_test, mix_min, mix_max, reverse=True),
    }

    bg_train = x_train[y_train == 0]
    cem_val = _cem_response(x_val, target_ref, bg_train if len(bg_train) > 8 else x_train)
    cem_test = _cem_response(x_test, target_ref, bg_train if len(bg_train) > 8 else x_train)
    cem_min, cem_max = float(cem_val.min()), float(cem_val.max())
    out["CEM"] = {
        "val": _norm_by_source(cem_val, cem_min, cem_max, reverse=False),
        "test": _norm_by_source(cem_test, cem_min, cem_max, reverse=False),
    }
    return out


def _build_ml_models(seed: int) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed)),
            ]
        ),
        "SVM-RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300, random_state=seed)),
            ]
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["XGBoost"] = Pipeline(
            [
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=seed,
                    ),
                )
            ]
        )
    except Exception:
        pass
    return models


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def _save_curve(curve_dir: str, model_name: str, seed: int, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    ensure_dir(curve_dir)
    file_path = os.path.join(curve_dir, f"{_safe_name(model_name)}_seed{seed}.npz")
    np.savez_compressed(file_path, y_true=np.asarray(y_true), y_prob=np.asarray(y_prob))


def _load_latest_curves(curve_dir: str, model_names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    model_curves: Dict[str, Dict[str, np.ndarray]] = {}
    for model_name in model_names:
        pattern = os.path.join(curve_dir, f"{_safe_name(model_name)}_seed*.npz")
        files = glob(pattern)
        if not files:
            continue
        latest = max(files, key=os.path.getmtime)
        data = np.load(latest)
        model_curves[model_name] = {
            "y_true": np.asarray(data["y_true"], dtype=np.int64),
            "y_prob": np.asarray(data["y_prob"], dtype=np.float32),
        }
    return model_curves


def _plot_metric_scatter(summary_df: pd.DataFrame, out_dir: str) -> None:
    if summary_df.empty:
        return
    ensure_dir(out_dir)
    x = summary_df["PR_AUC_mean"].values
    y = summary_df["F1_mean"].values
    mcc = summary_df["MCC_mean"].fillna(0.0).values
    size = np.clip(mcc, 0.0, 1.0) * 900.0 + 120.0
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=size, alpha=0.75, c="#4C78A8", edgecolors="black", linewidth=0.6)
    for _, row in summary_df.iterrows():
        plt.text(float(row["PR_AUC_mean"]) + 0.002, float(row["F1_mean"]) + 0.002, str(row["Model"]), fontsize=8)
    plt.xlabel("PR_AUC (mean)")
    plt.ylabel("F1 (mean)")
    plt.title("Comparative Trade-off (bubble size = MCC)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparative_tradeoff_scatter.png"), dpi=300)
    plt.close()


def _fit_calibrator(method: str, y_true: np.ndarray, y_prob: np.ndarray):
    if method == "platt":
        clf = LogisticRegression(max_iter=800)
        clf.fit(y_prob.reshape(-1, 1), y_true)
        return clf
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_prob.reshape(-1), y_true.reshape(-1))
        return iso
    return None


def _apply_calibrator(method: str, calibrator, y_prob: np.ndarray) -> np.ndarray:
    if calibrator is None or method == "none":
        return y_prob.astype(np.float32)
    if method == "platt":
        out = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        out = calibrator.predict(y_prob.reshape(-1))
    else:
        out = y_prob
    return np.asarray(out, dtype=np.float32)


def _best_threshold_from_validation(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    candidates = np.linspace(0.05, 0.95, 91)
    best_thr = 0.5
    best_score = -1.0
    for thr in candidates:
        pred = (y_prob >= thr).astype(np.int64)
        if metric == "mcc":
            tp = int(((pred == 1) & (y_true == 1)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            score = ((tp * tn - fp * fn) / np.sqrt(den + 1e-12)) if den > 0 else 0.0
        else:
            score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)
    return best_thr


def _refresh_reports(output_dir: str, detail_df: pd.DataFrame) -> None:
    if detail_df.empty:
        return

    detail_csv = os.path.join(output_dir, "comparative_metrics_detail.csv")
    summary_csv = os.path.join(output_dir, "comparative_metrics_summary.csv")
    leaderboard_csv = os.path.join(output_dir, "comparative_leaderboard.csv")
    significance_csv = os.path.join(output_dir, "comparative_significance.csv")
    curve_dir = os.path.join(output_dir, "curves")

    detail_df = detail_df.copy()
    dedup_cols = ["Seed", "Model"] if "Suite" not in detail_df.columns else ["Seed", "Suite", "Model"]
    detail_df = detail_df.drop_duplicates(subset=dedup_cols, keep="last")
    detail_df = detail_df.sort_values([c for c in ["Seed", "Model"] if c in detail_df.columns]).reset_index(drop=True)
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    agg_candidates = [
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "Kappa",
        "ROC_AUC",
        "PR_AUC",
        "Brier",
    ]
    agg_metrics = [c for c in agg_candidates if c in detail_df.columns]
    agg = detail_df.groupby("Model").agg({c: ["mean", "std"] for c in agg_metrics})
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    summary_df = agg.reset_index()
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    leaderboard = summary_df[["Model", "F1_mean", "PR_AUC_mean", "MCC_mean"]].copy()
    leaderboard["Rank_F1"] = leaderboard["F1_mean"].rank(ascending=False, method="min")
    leaderboard["Rank_PR_AUC"] = leaderboard["PR_AUC_mean"].rank(ascending=False, method="min")
    leaderboard["Rank_MCC"] = leaderboard["MCC_mean"].rank(ascending=False, method="min")
    leaderboard["Rank_Sum"] = leaderboard["Rank_F1"] + leaderboard["Rank_PR_AUC"] + leaderboard["Rank_MCC"]
    leaderboard = leaderboard.sort_values(["Rank_Sum", "F1_mean"], ascending=[True, False]).reset_index(drop=True)
    leaderboard.to_csv(leaderboard_csv, index=False, encoding="utf-8-sig")

    summary_for_plot = pd.DataFrame(
        {
            "Model": summary_df["Model"],
            "F1": summary_df["F1_mean"],
            "PR_AUC": summary_df["PR_AUC_mean"],
            "MCC": summary_df["MCC_mean"],
        }
    )
    plot_metric_bars(summary_for_plot, metrics=["F1", "PR_AUC", "MCC"], out_dir=output_dir, prefix="comparative")
    _plot_metric_scatter(summary_df=summary_df, out_dir=output_dir)

    model_curves = _load_latest_curves(curve_dir=curve_dir, model_names=summary_df["Model"].tolist())
    if model_curves:
        plot_roc_pr_curves(model_curves, out_dir=output_dir, prefix="comparative")
        if "AG-S2CNN" in model_curves:
            plot_probability_hist(
                y_true=model_curves["AG-S2CNN"]["y_true"],
                y_prob=model_curves["AG-S2CNN"]["y_prob"],
                out_path=os.path.join(output_dir, "ag_s2cnn_probability_hist.png"),
                title="AG-S2CNN Probability Histogram",
            )
        if "3D-CNN-Standard" in model_curves:
            plot_probability_hist(
                y_true=model_curves["3D-CNN-Standard"]["y_true"],
                y_prob=model_curves["3D-CNN-Standard"]["y_prob"],
                out_path=os.path.join(output_dir, "baseline_3dcnn_probability_hist.png"),
                title="3D-CNN Standard Probability Histogram",
            )

    significance_rows = []
    if "AG-S2CNN" in set(detail_df["Model"].tolist()):
        ref_keys = ["Seed", "F1", "PR_AUC", "MCC"]
        merge_keys = ["Seed"]
        ref_df = detail_df[detail_df["Model"] == "AG-S2CNN"][ref_keys].drop_duplicates(subset=merge_keys)
        for model_name in sorted(set(detail_df["Model"].tolist())):
            if model_name == "AG-S2CNN":
                continue
            cmp_df = detail_df[detail_df["Model"] == model_name][ref_keys].drop_duplicates(subset=merge_keys)
            merged = ref_df.merge(cmp_df, on=merge_keys, suffixes=("_ref", "_cmp"))
            if len(merged) < 2:
                continue
            for metric_name in ["F1", "PR_AUC", "MCC"]:
                test = paired_significance(
                    metric_values_ref=merged[f"{metric_name}_ref"].tolist(),
                    metric_values_cmp=merged[f"{metric_name}_cmp"].tolist(),
                    prefer="higher",
                )
                significance_rows.append(
                    {
                        "Reference": "AG-S2CNN",
                        "Compared": model_name,
                        "Metric": metric_name,
                        "Test": test["test"],
                        "p_value": test["p_value"],
                        "effect": test["effect"],
                    }
                )
    pd.DataFrame(significance_rows).to_csv(significance_csv, index=False, encoding="utf-8-sig")

    save_json(
        {
            "detail_csv": detail_csv,
            "summary_csv": summary_csv,
            "leaderboard_csv": leaderboard_csv,
            "significance_csv": significance_csv,
            "figures": [
                "comparative_tradeoff_scatter.png",
                "comparative_roc_curves.png",
                "comparative_pr_curves.png",
                "comparative_F1_bar.png",
                "comparative_PR_AUC_bar.png",
                "comparative_MCC_bar.png",
            ],
            "note": "当前脚本按单方案执行并持续增量更新报告，便于快速获得阶段性结果。",
        },
        os.path.join(output_dir, "comparative_report_index.json"),
    )


def _select_scheme(suite: str, scheme: str) -> str:
    physical_models = ["SAM", "SID", "SAM-SID", "CEM"]
    ml_models = list(_build_ml_models(seed=2025).keys())
    deep_models = list(get_comparative_deep_builders().keys())

    model_map = {
        "physical": physical_models,
        "ml": ml_models,
        "deep": deep_models,
    }
    defaults = {"physical": "SAM-SID", "ml": "RandomForest", "deep": "AG-S2CNN"}

    if scheme == "auto":
        return defaults[suite]
    if scheme not in model_map[suite]:
        raise ValueError(f"suite={suite} 不支持方案 {scheme}。可选：{model_map[suite]}")
    return scheme


def main() -> None:
    parser = argparse.ArgumentParser(description="运行模型对比实验（单方案增量执行版）")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="D:/Grp_data/Air-Ground_Spectral-Spatial_CNN/3_model/evaluation_outputs/comparative",
    )
    parser.add_argument("--suite", type=str, choices=["physical", "ml", "deep"], default="deep")
    parser.add_argument("--scheme", type=str, default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--deep-opt-threshold", action="store_true")
    parser.add_argument("--deep-threshold-metric", type=str, choices=["f1", "mcc"], default="f1")
    parser.add_argument("--deep-calibration", type=str, choices=["none", "platt", "isotonic"], default="none")
    parser.add_argument("--deep-tta", action="store_true")
    parser.add_argument("--deep-boost", action="store_true", help="启用训练强化包（组合损失+Mixup/CutMix+EMA/SWA+CosineLR）")
    parser.add_argument("--deep-ema", type=int, choices=[0, 1], default=1)
    parser.add_argument("--deep-swa", type=int, choices=[0, 1], default=1)
    parser.add_argument("--deep-mixup", type=int, choices=[0, 1], default=1)
    parser.add_argument("--deep-cutmix", type=int, choices=[0, 1], default=1)
    parser.add_argument("--deep-cosine-lr", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    output_dir = args.output_dir
    ensure_dir(output_dir)
    curve_dir = os.path.join(output_dir, "curves")
    ensure_dir(curve_dir)

    selected_scheme = _select_scheme(args.suite, args.scheme)
    detail_csv = os.path.join(output_dir, "comparative_metrics_detail.csv")
    if os.path.exists(detail_csv):
        detail_df = pd.read_csv(detail_csv, encoding="utf-8-sig")
    else:
        detail_df = pd.DataFrame()

    print(f"[开始] 对比实验 suite={args.suite}, scheme={selected_scheme}, seeds={args.seeds}")
    for idx, seed in enumerate(args.seeds, start=1):
        print(f"[进度] seed {idx}/{len(args.seeds)} -> {seed}")
        set_global_seed(seed)
        exp = load_experiment_data(config_path=args.config, seed=seed, train_ratio=0.6, val_ratio=0.2)
        bands = exp.image_cube.shape[2]
        spatial_size = int(exp.config.get("model.spatial_size", 25))

        x_train_c, y_train = extract_center_spectra(exp.image_cube, exp.train_samples)
        x_val_c, y_val = extract_center_spectra(exp.image_cube, exp.val_samples)
        x_test_c, y_test = extract_center_spectra(exp.image_cube, exp.test_samples)
        x_train_stat, _ = extract_spectral_and_local_stats(exp.image_cube, exp.train_samples, local_radius=1)
        x_val_stat, _ = extract_spectral_and_local_stats(exp.image_cube, exp.val_samples, local_radius=1)
        x_test_stat, _ = extract_spectral_and_local_stats(exp.image_cube, exp.test_samples, local_radius=1)

        target_ref = exp.mapped_gsrsl[exp.target_class_index].astype(np.float32)
        model_name = selected_scheme
        thr = 0.5

        if args.suite == "physical":
            phy_probs = _run_physical_methods(x_train_c, y_train, x_val_c, y_val, x_test_c, target_ref)
            y_val_prob = phy_probs[model_name]["val"].astype(np.float32)
            y_prob = phy_probs[model_name]["test"].astype(np.float32)
        elif args.suite == "ml":
            ml_models = _build_ml_models(seed=seed)
            model = ml_models[model_name]
            use_stat = model_name in {"SVM-RBF", "RandomForest", "MLP", "XGBoost"}
            x_tr = x_train_stat if use_stat else x_train_c
            x_va = x_val_stat if use_stat else x_val_c
            x_te = x_test_stat if use_stat else x_test_c
            model.fit(x_tr, y_train)
            if hasattr(model, "predict_proba"):
                y_val_prob = model.predict_proba(x_va)[:, 1].astype(np.float32)
                y_prob = model.predict_proba(x_te)[:, 1].astype(np.float32)
            else:
                score_val = model.decision_function(x_va)
                score = model.decision_function(x_te)
                y_val_prob = ((score_val - score_val.min()) / (score_val.max() - score_val.min() + 1e-12)).astype(np.float32)
                y_prob = ((score - score.min()) / (score.max() - score.min() + 1e-12)).astype(np.float32)
        else:
            deep_builders = get_comparative_deep_builders()
            device = torch.device("cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

            train_ds = build_dataset(exp, exp.train_samples, augmentation=True)
            val_ds = build_dataset(exp, exp.val_samples, augmentation=False)
            test_ds = build_dataset(exp, exp.test_samples, augmentation=False)
            train_labels = np.array([s[2] for s in exp.train_samples], dtype=np.int64)
            bg_count = max(1.0, float((train_labels == 0).sum()))
            pos_count = max(1.0, float((train_labels == 1).sum()))
            pos_weight = bg_count / pos_count
            loss_cfg = {
                "hnm_ratio": float(exp.config.get("training.hnm_ratio", 0.5)),
                "hnm_weight": float(exp.config.get("training.hnm_weight", 0.2)),
                "separation_weight": float(exp.config.get("training.separation_weight", 0.2)),
                "separation_margin": float(exp.config.get("training.separation_margin", 0.06)),
            }
            model = deep_builders[model_name](bands, spatial_size)
            boost_enabled = bool(args.deep_boost)
            boost_cfg = {
                "use_ema": bool(args.deep_ema),
                "use_swa": bool(args.deep_swa),
                "use_mixup": bool(args.deep_mixup),
                "use_cutmix": bool(args.deep_cutmix),
                "use_cosine_lr": bool(args.deep_cosine_lr),
            }
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
                pos_weight=float(pos_weight),
                early_stopping_patience=int(exp.config.get("training.early_stopping_patience", 8)),
                verbose=True,
                log_prefix=f"[Deep][{model_name}][seed={seed}]",
                boost=boost_enabled,
                boost_cfg=boost_cfg,
            )
            pred_val = predict_binary_deep_model(
                model=trained["model"],
                dataset=val_ds,
                device=device,
                batch_size=int(exp.config.get("training.batch_size", 8)),
                num_workers=int(exp.config.get("training.num_workers", 0)),
                tta=(args.deep_tta and model_name == "AG-S2CNN"),
            )
            pred = predict_binary_deep_model(
                model=trained["model"],
                dataset=test_ds,
                device=device,
                batch_size=int(exp.config.get("training.batch_size", 8)),
                num_workers=int(exp.config.get("training.num_workers", 0)),
                tta=(args.deep_tta and model_name == "AG-S2CNN"),
            )
            y_prob = pred["y_prob"].astype(np.float32)
            y_val_prob = pred_val["y_prob"].astype(np.float32)

            if model_name == "AG-S2CNN" and args.deep_calibration != "none":
                calibrator = _fit_calibrator(args.deep_calibration, pred_val["y_true"], y_val_prob)
                y_val_prob = _apply_calibrator(args.deep_calibration, calibrator, y_val_prob)
                y_prob = _apply_calibrator(args.deep_calibration, calibrator, y_prob)
            if model_name == "AG-S2CNN" and args.deep_opt_threshold:
                thr = _best_threshold_from_validation(
                    y_true=pred_val["y_true"],
                    y_prob=y_val_prob,
                    metric=args.deep_threshold_metric,
                )

        eval_threshold = thr if (args.suite == "deep" and model_name == "AG-S2CNN") else 0.5
        metrics = compute_binary_metrics(y_true=y_test, y_prob=y_prob, threshold=eval_threshold)

        new_row = {
            "Seed": seed,
            "Suite": args.suite,
            "Model": model_name,
        }
        if args.suite == "deep" and model_name == "AG-S2CNN":
            new_row["EvalThreshold"] = float(eval_threshold)
            new_row["Calibration"] = args.deep_calibration
            new_row["TTA"] = bool(args.deep_tta)
            new_row["Boost"] = bool(args.deep_boost)
        new_row.update(metrics)
        detail_df = pd.concat([detail_df, pd.DataFrame([new_row])], ignore_index=True)

        _save_curve(curve_dir=curve_dir, model_name=model_name, seed=seed, y_true=y_test, y_prob=y_prob)
        _refresh_reports(output_dir=output_dir, detail_df=detail_df)
        print(f"[结果] seed={seed}, model={model_name}, F1={metrics['F1']:.4f}, PR_AUC={metrics['PR_AUC']:.4f}, MCC={metrics['MCC']:.4f}")

    print(f"[完成] 对比实验输出目录: {os.path.abspath(output_dir)}")
    print("[提示] 下一次运行请切换 --scheme，以增量方式继续补齐其它对比方案。")


if __name__ == "__main__":
    main()
