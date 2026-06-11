import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from train import (
    _apply_band_subset,
    _build_stratified_samples,
    _build_target_gsrsl,
    _compute_binary_enhanced_loss,
    _get_label_schema,
    _normalize_gsrsl,
    _remap_ground_truth_labels,
    _split_samples,
    _split_samples_non_overlapping,
)
from utils.dataset import HyperspectralDataset, load_hyperspectral_data


@dataclass
class ExperimentData:
    config: Config
    image_cube: np.ndarray
    mapped_ground_truth: np.ndarray
    mapped_gsrsl: Dict[int, np.ndarray]
    train_samples: List[Tuple[int, int, int]]
    val_samples: List[Tuple[int, int, int]]
    test_samples: List[Tuple[int, int, int]]
    class_names: List[str]
    target_class_index: int
    background_class_index: int
    metadata: Dict[str, Any]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_experiment_data(
    config_path: str,
    seed: int = 2025,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    spatial_size_override: Optional[int] = None,
) -> ExperimentData:
    config = Config.from_yaml(config_path) if os.path.exists(config_path) else Config()

    image_path = config.get("data.image_path", "")
    gt_path = config.get("data.ground_truth_path", "")
    gsrsl_path = config.get("data.gsrsl_path", "")

    image_cube, ground_truth_raw, gsrsl_raw, metadata = load_hyperspectral_data(
        image_path=image_path,
        gt_path=gt_path,
        gsrsl_path=gsrsl_path,
        return_metadata=True,
    )
    image_cube = np.asarray(image_cube, dtype=np.float32)
    ground_truth_raw = np.asarray(ground_truth_raw)

    band_keep_indices = config.get("data.band_keep_indices", None)
    if band_keep_indices is not None:
        gsrsl_raw_dict = _normalize_gsrsl(gsrsl_raw)
        image_cube, gsrsl_raw_dict = _apply_band_subset(
            image_cube=image_cube,
            gsrsl=gsrsl_raw_dict,
            keep_indices=np.asarray(band_keep_indices, dtype=np.int64),
        )
        gsrsl_raw = gsrsl_raw_dict

    inferred_bands = int(image_cube.shape[2])
    config.config["model"]["num_bands"] = inferred_bands
    config.config["model"]["num_classes"] = 1
    if spatial_size_override is not None:
        config.config["model"]["spatial_size"] = int(spatial_size_override)

    label_schema = _get_label_schema(config)
    class_names = label_schema["class_names"]
    target_class_index = int(label_schema.get("target_class_index", 1))
    background_class_index = int(label_schema["background_class_index"])

    mapped_ground_truth = _remap_ground_truth_labels(
        ground_truth=ground_truth_raw,
        schema=label_schema,
        num_classes=2,
    )
    gsrsl_dict = _normalize_gsrsl(gsrsl_raw)
    mapped_gsrsl = _build_target_gsrsl(gsrsl_dict, label_schema, inferred_bands)

    by_class = _build_stratified_samples(
        mapped_ground_truth=mapped_ground_truth,
        num_classes=2,
        background_class_index=background_class_index,
        background_sample_ratio=float(config.get("training.background_sample_ratio", 5.0)),
        seed=seed,
        max_samples_per_class=config.get("training.max_samples_per_class", None),
    )
    spatial_size = int(config.get("model.spatial_size", 25))

    train_samples, val_samples, test_samples = _split_samples_non_overlapping(
        mapped_ground_truth=mapped_ground_truth,
        by_class=by_class,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        spatial_size=spatial_size,
        seed=seed,
    )
    if len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
        train_samples, val_samples, test_samples = _split_samples(by_class, train_ratio, val_ratio)

    return ExperimentData(
        config=config,
        image_cube=image_cube,
        mapped_ground_truth=mapped_ground_truth,
        mapped_gsrsl=mapped_gsrsl,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        class_names=class_names,
        target_class_index=target_class_index,
        background_class_index=background_class_index,
        metadata=metadata,
    )


def build_dataset(exp: ExperimentData, samples: List[Tuple[int, int, int]], augmentation: bool) -> HyperspectralDataset:
    return HyperspectralDataset(
        image_cube=exp.image_cube,
        ground_truth=exp.mapped_ground_truth,
        gsrsl=exp.mapped_gsrsl,
        spatial_size=int(exp.config.get("model.spatial_size", 25)),
        augmentation=augmentation,
        sample_list=samples,
    )


def sample_labels(samples: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    if len(samples) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.array([s[2] for s in samples], dtype=np.int64)


def extract_center_spectra(
    image_cube: np.ndarray,
    samples: Sequence[Tuple[int, int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    feats = []
    labels = []
    for i, j, lab in samples:
        feats.append(image_cube[int(i), int(j), :].astype(np.float32))
        labels.append(int(lab))
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def extract_spectral_and_local_stats(
    image_cube: np.ndarray,
    samples: Sequence[Tuple[int, int, int]],
    local_radius: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = image_cube.shape
    feats = []
    labels = []
    for i, j, lab in samples:
        i0 = max(0, i - local_radius)
        i1 = min(h, i + local_radius + 1)
        j0 = max(0, j - local_radius)
        j1 = min(w, j + local_radius + 1)
        patch = image_cube[i0:i1, j0:j1, :]
        mean_v = patch.mean(axis=(0, 1))
        std_v = patch.std(axis=(0, 1))
        center = image_cube[i, j, :]
        feats.append(np.concatenate([center, mean_v, std_v], axis=0).astype(np.float32))
        labels.append(int(lab))
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    oa = float((tp + tn) / max(1, tp + tn + fp + fn))
    ba = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    po = oa
    total = max(1, tp + tn + fp + fn)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / float(total * total)
    kappa = (po - pe) / (1 - pe + 1e-12)
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_roc = 0.0
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0
    brier = float(np.mean((y_prob - y_true.astype(np.float32)) ** 2))

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "OA": float(oa),
        "BalancedAccuracy": float(ba),
        "MCC": float(mcc),
        "Kappa": float(kappa),
        "ROC_AUC": float(auc_roc),
        "PR_AUC": float(pr_auc),
        "Brier": float(brier),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def connected_components(binary_map: np.ndarray) -> List[int]:
    binary = np.asarray(binary_map).astype(np.uint8)
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=np.uint8)
    areas: List[int] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(h):
        for j in range(w):
            if binary[i, j] == 0 or visited[i, j] == 1:
                continue
            stack = [(i, j)]
            visited[i, j] = 1
            area = 0
            while stack:
                x, y = stack.pop()
                area += 1
                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and binary[nx, ny] == 1 and visited[nx, ny] == 0:
                        visited[nx, ny] = 1
                        stack.append((nx, ny))
            areas.append(area)
    return areas


def compute_spatial_metrics(
    probability_map: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.6,
) -> Dict[str, float]:
    prob = np.asarray(probability_map, dtype=np.float32)
    gt = np.asarray(gt_mask).astype(bool)
    pred = prob >= float(threshold)
    areas = connected_components(pred.astype(np.uint8))
    n_cc = len(areas)
    a_mean = float(np.mean(areas)) if areas else 0.0
    coverage_main = float(np.logical_and(pred, gt).sum() / (gt.sum() + 1e-12))
    false_alarm = float(np.logical_and(pred, ~gt).sum() / (pred.sum() + 1e-12))
    gx, gy = np.gradient(prob)
    sharpness = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    return {
        "N_cc": float(n_cc),
        "A_mean": a_mean,
        "Coverage_main": coverage_main,
        "FalseAlarm_area_ratio": false_alarm,
        "Sharpness": sharpness,
    }


def train_binary_deep_model(
    model: nn.Module,
    train_dataset: HyperspectralDataset,
    val_dataset: HyperspectralDataset,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    weight_decay: float,
    loss_cfg: Dict[str, float],
    pos_weight: float,
    early_stopping_patience: int = 8,
    verbose: bool = False,
    log_prefix: str = "",
    boost: bool = False,
    boost_cfg: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    boost_cfg = dict(boost_cfg or {})
    use_mixup = bool(boost_cfg.get("use_mixup", True))
    use_cutmix = bool(boost_cfg.get("use_cutmix", True))
    use_aux_loss = bool(boost_cfg.get("use_aux_loss", True))
    use_ema = bool(boost_cfg.get("use_ema", True))
    use_swa = bool(boost_cfg.get("use_swa", True))
    use_cosine_lr = bool(boost_cfg.get("use_cosine_lr", True))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    ) if boost else torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = None
    if boost and use_cosine_lr and int(num_epochs) > 1:
        eta_min = float(learning_rate) * float(boost_cfg.get("eta_min_ratio", 0.05))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(num_epochs), eta_min=eta_min)

    best_state = None
    best_score = -1.0
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_pr_auc": []}
    ema_state: Optional[Dict[str, torch.Tensor]] = None
    ema_decay = float(boost_cfg.get("ema_decay", 0.999))
    last_states: List[Dict[str, torch.Tensor]] = []
    swa_keep = int(boost_cfg.get("swa_keep", 5))

    def _binary_focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        mod = (1.0 - p_t).pow(gamma)
        return (alpha_t * mod * bce).mean()

    def _binary_tversky_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.3, beta: float = 0.7) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()
        tv = (tp + 1e-8) / (tp + alpha * fp + beta * fn + 1e-8)
        return 1.0 - tv

    for _epoch in range(1, int(num_epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        for x_sat, x_ref, y in train_loader:
            x_sat = x_sat.to(device)
            x_ref = x_ref.to(device)
            y = y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            if boost:
                y = y.view(-1)
                # MixUp for robustness.
                mixup_alpha = float(boost_cfg.get("mixup_alpha", 0.2))
                if use_mixup and mixup_alpha > 0 and np.random.rand() < float(boost_cfg.get("mixup_prob", 0.6)):
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    perm = torch.randperm(x_sat.size(0), device=x_sat.device)
                    x_sat = lam * x_sat + (1.0 - lam) * x_sat[perm]
                    x_ref = lam * x_ref + (1.0 - lam) * x_ref[perm]
                    y = lam * y + (1.0 - lam) * y[perm]

                # CutMix on the spatial branch.
                cutmix_prob = float(boost_cfg.get("cutmix_prob", 0.2))
                if use_cutmix and cutmix_prob > 0 and np.random.rand() < cutmix_prob and x_sat.size(0) > 1:
                    perm = torch.randperm(x_sat.size(0), device=x_sat.device)
                    _, _, _, h, w = x_sat.shape
                    ratio = float(np.random.uniform(0.25, 0.5))
                    cut_w = max(1, int(w * ratio))
                    cut_h = max(1, int(h * ratio))
                    cx = int(np.random.randint(0, w))
                    cy = int(np.random.randint(0, h))
                    x1 = max(0, cx - cut_w // 2)
                    y1 = max(0, cy - cut_h // 2)
                    x2 = min(w, x1 + cut_w)
                    y2 = min(h, y1 + cut_h)
                    x_sat[:, :, :, y1:y2, x1:x2] = x_sat[perm, :, :, y1:y2, x1:x2]
                    lam_adj = 1.0 - float((x2 - x1) * (y2 - y1)) / float(max(1, w * h))
                    y = lam_adj * y + (1.0 - lam_adj) * y[perm]

            logits = model(x_sat, x_ref)
            loss, _ = _compute_binary_enhanced_loss(
                logits=logits,
                targets=y,
                criterion=criterion,
                hnm_ratio=float(loss_cfg.get("hnm_ratio", 0.5)),
                hnm_weight=float(loss_cfg.get("hnm_weight", 0.2)),
                separation_weight=float(loss_cfg.get("separation_weight", 0.2)),
                separation_margin=float(loss_cfg.get("separation_margin", 0.06)),
            )
            if boost and use_aux_loss:
                logits_vec = logits.view(-1)
                targets_vec = y.float().view(-1)
                focal = _binary_focal_loss(
                    logits_vec,
                    targets_vec,
                    gamma=float(boost_cfg.get("focal_gamma", 2.0)),
                    alpha=float(boost_cfg.get("focal_alpha", 0.35)),
                )
                tversky = _binary_tversky_loss(
                    logits_vec,
                    targets_vec,
                    alpha=float(boost_cfg.get("tversky_alpha", 0.3)),
                    beta=float(boost_cfg.get("tversky_beta", 0.7)),
                )
                loss = (
                    loss
                    + float(boost_cfg.get("focal_weight", 0.35)) * focal
                    + float(boost_cfg.get("tversky_weight", 0.30)) * tversky
                )
            loss.backward()
            optimizer.step()
            if boost and use_ema:
                state_now = model.state_dict()
                if ema_state is None:
                    ema_state = {k: v.detach().cpu().clone() for k, v in state_now.items()}
                else:
                    for k, v in state_now.items():
                        v_cpu = v.detach().cpu()
                        if torch.is_floating_point(v_cpu):
                            ema_state[k].mul_(ema_decay).add_(v_cpu, alpha=(1.0 - ema_decay))
                        else:
                            ema_state[k] = v_cpu.clone()
            train_loss_sum += float(loss.item())

        if scheduler is not None:
            scheduler.step()

        if boost and use_swa:
            last_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if len(last_states) > max(2, swa_keep):
                last_states.pop(0)

        train_loss = train_loss_sum / max(1, len(train_loader))
        restore_state = None
        if boost and use_ema and ema_state is not None:
            restore_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: v.to(device) for k, v in ema_state.items()})
        val_result = predict_binary_deep_model(model, val_dataset, device, batch_size, num_workers)
        val_metrics = compute_binary_metrics(val_result["y_true"], val_result["y_prob"], threshold=0.5)
        val_loss = float(
            nn.BCEWithLogitsLoss()(
                torch.from_numpy(np.log(np.clip(val_result["y_prob"], 1e-8, 1.0 - 1e-8) / (1.0 - np.clip(val_result["y_prob"], 1e-8, 1.0 - 1e-8)))).float(),
                torch.from_numpy(val_result["y_true"]).float(),
            ).item()
        )
        if restore_state is not None:
            model.load_state_dict(restore_state)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["F1"])
        history["val_pr_auc"].append(val_metrics["PR_AUC"])

        score = 0.5 * val_metrics["F1"] + 0.5 * val_metrics["PR_AUC"]
        if score > best_score:
            best_score = score
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            tag = "best"
        else:
            bad_epochs += 1
            tag = "keep"
        if verbose:
            prefix = f"{log_prefix} " if log_prefix else ""
            print(
                f"{prefix}[Epoch {_epoch}/{int(num_epochs)}] "
                f"train_loss={train_loss:.4f} val_f1={val_metrics['F1']:.4f} "
                f"val_pr_auc={val_metrics['PR_AUC']:.4f} best={best_score:.4f} "
                f"patience={bad_epochs}/{int(early_stopping_patience)} ({tag})",
                flush=True,
            )
        if bad_epochs >= int(early_stopping_patience):
            if verbose:
                prefix = f"{log_prefix} " if log_prefix else ""
                print(f"{prefix}[EarlyStop] epoch={_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif boost and use_swa and len(last_states) >= 2:
        avg_state = {}
        for k in last_states[0].keys():
            avg_state[k] = torch.stack([s[k] for s in last_states], dim=0).mean(dim=0)
        model.load_state_dict({k: v.to(device) for k, v in avg_state.items()})

    return {"model": model, "history": history, "best_score": float(best_score)}


def predict_binary_deep_model(
    model: nn.Module,
    dataset: HyperspectralDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    tta: bool = False,
    tta_modes: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []

    modes = list(tta_modes) if tta_modes is not None else ["none", "flip_h", "flip_v", "flip_hv"]

    def _apply_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "flip_h":
            return torch.flip(x, dims=[-1])
        if mode == "flip_v":
            return torch.flip(x, dims=[-2])
        if mode == "flip_hv":
            return torch.flip(x, dims=[-1, -2])
        return x

    with torch.no_grad():
        for x_sat, x_ref, y in loader:
            x_sat = x_sat.to(device)
            x_ref = x_ref.to(device)
            if tta:
                probs = []
                for mode in modes:
                    logits = model(_apply_mode(x_sat, mode), _apply_mode(x_ref, mode))
                    probs.append(torch.sigmoid(logits).view(-1))
                prob_t = torch.stack(probs, dim=0).mean(dim=0)
            else:
                logits = model(x_sat, x_ref)
                prob_t = torch.sigmoid(logits).view(-1)
            prob = prob_t.detach().cpu().numpy()
            y_prob.extend(prob.tolist())
            y_true.extend(y.numpy().reshape(-1).tolist())
    return {
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_prob": np.asarray(y_prob, dtype=np.float32),
    }


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 2026) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape[0] != b.shape[0] or a.shape[0] == 0:
        return 1.0
    diff = a - b
    obs = abs(float(diff.mean()))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(n_perm)):
        signs = rng.choice([-1.0, 1.0], size=diff.shape[0])
        stat = abs(float((diff * signs).mean()))
        if stat >= obs:
            count += 1
    return float((count + 1) / (n_perm + 1))


def paired_significance(
    metric_values_ref: Sequence[float],
    metric_values_cmp: Sequence[float],
    prefer: str = "higher",
) -> Dict[str, Any]:
    ref = np.asarray(metric_values_ref, dtype=np.float64)
    cmp = np.asarray(metric_values_cmp, dtype=np.float64)
    if ref.shape != cmp.shape or ref.size == 0:
        return {"test": "invalid", "p_value": 1.0, "effect": 0.0}
    effect = float(np.mean(ref - cmp) if prefer == "higher" else np.mean(cmp - ref))

    try:
        from scipy.stats import ttest_rel  # type: ignore

        stat = ttest_rel(ref, cmp, nan_policy="omit")
        p_value = float(stat.pvalue if stat.pvalue is not None else 1.0)
        test_name = "paired_t_test"
    except Exception:
        p_value = permutation_pvalue(ref, cmp, n_perm=4000)
        test_name = "paired_permutation_sign_flip"

    return {"test": test_name, "p_value": p_value, "effect": effect}


def plot_roc_pr_curves(
    model_results: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
    prefix: str = "comparative",
) -> None:
    ensure_dir(out_dir)
    plt.figure(figsize=(10, 6))
    for name, result in model_results.items():
        y_true = result["y_true"]
        y_prob = result["y_prob"]
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=2)
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_roc_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, result in model_results.items():
        y_true = result["y_true"]
        y_prob = result["y_prob"]
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})", linewidth=2)
        except Exception:
            continue
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_pr_curves.png"), dpi=300)
    plt.close()


def plot_metric_bars(
    metric_table: pd.DataFrame,
    metrics: Sequence[str],
    out_dir: str,
    prefix: str = "comparative",
) -> None:
    ensure_dir(out_dir)
    table = metric_table.copy()
    for m in metrics:
        if m not in table.columns:
            continue
        plt.figure(figsize=(12, 5))
        order = table.sort_values(m, ascending=False)
        plt.bar(order["Model"], order[m], color="#4C78A8")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(m)
        plt.title(f"{m} Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_{m}_bar.png"), dpi=300)
        plt.close()


def plot_probability_hist(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    plt.figure(figsize=(9, 5))
    plt.hist(neg, bins=40, alpha=0.6, density=True, label="Background")
    plt.hist(pos, bins=40, alpha=0.6, density=True, label="Target")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=300)
    plt.close()
