import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

from config import Config
from models.ag_s2cnn import AG_S2CNN
from train import load_checkpoint
from utils.dataset import _load_raster_array


def _get_device(use_cuda: bool = True, gpu_id: int = 0) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def _configure_chinese_font() -> str:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "SimSun",
        "KaiTi",
        "Arial Unicode MS",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    selected = [name for name in candidates if name in available]

    if selected:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
        chosen = selected[0]
    else:
        chosen = "DejaVu Sans"

    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def _extract_patch(image_cube: np.ndarray, center_row: int, center_col: int, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    row_start = center_row - half
    row_end = center_row + half + 1
    col_start = center_col - half
    col_end = center_col + half + 1
    height, width, _ = image_cube.shape

    if row_start < 0 or row_end > height or col_start < 0 or col_end > width:
        pad_top = max(0, -row_start)
        pad_bottom = max(0, row_end - height)
        pad_left = max(0, -col_start)
        pad_right = max(0, col_end - width)
        valid_row_start = max(0, row_start)
        valid_row_end = min(height, row_end)
        valid_col_start = max(0, col_start)
        valid_col_end = min(width, col_end)
        valid_patch = image_cube[valid_row_start:valid_row_end, valid_col_start:valid_col_end, :]
        return np.pad(
            valid_patch,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect",
        )

    return image_cube[row_start:row_end, col_start:col_end, :]


def _parse_rgb_bands(rgb_bands: List[int], num_bands: int) -> List[int]:
    if rgb_bands:
        parsed = [int(v) for v in rgb_bands]
        if len(parsed) != 3:
            raise ValueError("--rgb-bands 必须提供 3 个整数")
        if min(parsed) >= 1 and max(parsed) <= num_bands:
            parsed = [v - 1 for v in parsed]
        for value in parsed:
            if value < 0 or value >= num_bands:
                raise ValueError(f"RGB 波段越界: {value}")
        return parsed

    auto_bands = np.linspace(0, num_bands - 1, 5, dtype=np.int32)[1:4]
    return auto_bands.tolist()


def _normalize_to_unit(arr: np.ndarray, low_q: float = 2.0, high_q: float = 98.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lower = float(np.nanpercentile(arr, low_q))
    upper = float(np.nanpercentile(arr, high_q))
    if upper <= lower:
        return np.zeros_like(arr, dtype=np.float32)
    normalized = (arr - lower) / (upper - lower)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _build_pseudocolor(patch_cube: np.ndarray, rgb_bands: List[int]) -> np.ndarray:
    rgb = patch_cube[:, :, rgb_bands].astype(np.float32)
    return _normalize_to_unit(rgb)


def _project_channel_to_map(channel_volume: np.ndarray) -> np.ndarray:
    channel_volume = np.asarray(channel_volume, dtype=np.float32)
    projection = np.maximum(channel_volume, 0.0).mean(axis=0)
    return _normalize_to_unit(projection)


def _local_mean_3x3(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr, ((1, 1), (1, 1)), mode="reflect")
    result = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0
    return result.astype(np.float32)


def _score_feature_map(feature_map: np.ndarray) -> Dict[str, float]:
    gy, gx = np.gradient(feature_map)
    edge_score = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

    smoothed = _local_mean_3x3(feature_map)
    texture_score = float(np.mean(np.abs(feature_map - smoothed)))

    peak_region = feature_map[feature_map >= np.quantile(feature_map, 0.99)]
    if peak_region.size == 0:
        peak_region = feature_map.reshape(-1)
    anomaly_score = float(np.mean(peak_region) - np.mean(feature_map))

    total_score = edge_score + texture_score + anomaly_score
    return {
        "edge": edge_score,
        "texture": texture_score,
        "anomaly": anomaly_score,
        "total": total_score,
    }


def _select_representative_channels(
    feature_tensor: torch.Tensor,
    labels: List[str],
) -> List[Dict[str, object]]:
    feature_np = feature_tensor.detach().cpu().numpy()[0]
    candidates = []
    for channel_idx in range(feature_np.shape[0]):
        feature_map = _project_channel_to_map(feature_np[channel_idx])
        scores = _score_feature_map(feature_map)
        candidates.append(
            {
                "channel_idx": int(channel_idx),
                "feature_map": feature_map,
                "scores": scores,
            }
        )

    selected: List[Dict[str, object]] = []
    used = set()
    for label_name, metric_name in zip(labels, ["edge", "texture", "anomaly"]):
        sorted_candidates = sorted(
            candidates,
            key=lambda item: item["scores"][metric_name],
            reverse=True,
        )
        chosen = None
        for candidate in sorted_candidates:
            if candidate["channel_idx"] not in used:
                chosen = candidate
                break
        if chosen is None:
            chosen = sorted_candidates[0]
        used.add(chosen["channel_idx"])
        selected.append(
            {
                "label": label_name,
                "channel_idx": chosen["channel_idx"],
                "feature_map": chosen["feature_map"],
                "scores": chosen["scores"],
            }
        )
    return selected


def _select_fused_channel(feature_tensor: torch.Tensor) -> Dict[str, object]:
    feature_np = feature_tensor.detach().cpu().numpy()[0]
    candidates = []
    for channel_idx in range(feature_np.shape[0]):
        feature_map = _project_channel_to_map(feature_np[channel_idx])
        scores = _score_feature_map(feature_map)
        candidates.append(
            {
                "channel_idx": int(channel_idx),
                "feature_map": feature_map,
                "scores": scores,
            }
        )

    best = max(candidates, key=lambda item: item["scores"]["total"])
    return {
        "label": "Representative fused activation",
        "channel_idx": best["channel_idx"],
        "feature_map": best["feature_map"],
        "scores": best["scores"],
    }


def _score_branch_tensor(feature_tensor: torch.Tensor) -> Dict[str, object]:
    feature_np = feature_tensor.detach().cpu().numpy()[0]
    aggregate_map = _normalize_to_unit(np.maximum(feature_np, 0.0).mean(axis=(0, 1)))
    scores = _score_feature_map(aggregate_map)
    active_region = aggregate_map[aggregate_map >= np.quantile(aggregate_map, 0.95)]
    if active_region.size == 0:
        active_region = aggregate_map.reshape(-1)
    saliency_score = float(np.mean(active_region))
    total_score = float(scores["total"] + saliency_score)
    return {
        "aggregate_map": aggregate_map,
        "scores": scores,
        "saliency_score": saliency_score,
        "total_score": total_score,
    }


def _score_input_patch(pseudocolor: np.ndarray) -> float:
    gray = np.asarray(pseudocolor, dtype=np.float32).mean(axis=2)
    gy, gx = np.gradient(gray)
    gradient_score = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
    texture_score = float(np.std(gray))
    return gradient_score + texture_score


def _run_backbone(model: AG_S2CNN, patch_cube: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    patch_tensor = torch.from_numpy(np.transpose(patch_cube, (2, 0, 1))).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        reduced = model.s_backbone.spectral_reduction(patch_tensor)
        small_scale = model.s_backbone.inception.branch_local(reduced)
        large_scale = model.s_backbone.inception.branch_context(reduced)
        concatenated = torch.cat([small_scale, large_scale], dim=1)
        mapped = model.s_backbone.channel_mapping(concatenated)
    return reduced, small_scale, large_scale, concatenated, mapped


def _find_representative_patch(
    image_cube: np.ndarray,
    model: AG_S2CNN,
    device: torch.device,
    patch_size: int,
    rgb_bands: List[int],
    search_stride: int,
) -> Tuple[int, int, Dict[str, float]]:
    height, width, _ = image_cube.shape
    half = patch_size // 2
    row_candidates = list(range(half, max(half + 1, height - half), search_stride))
    col_candidates = list(range(half, max(half + 1, width - half), search_stride))

    if row_candidates[-1] != height - half - 1 and height - half - 1 >= 0:
        row_candidates.append(height - half - 1)
    if col_candidates[-1] != width - half - 1 and width - half - 1 >= 0:
        col_candidates.append(width - half - 1)

    best_row = height // 2
    best_col = width // 2
    best_info = {
        "input_score": -1.0,
        "small_score": -1.0,
        "large_score": -1.0,
        "total_score": -1.0,
    }

    for row in row_candidates:
        for col in col_candidates:
            patch_cube = _extract_patch(image_cube, row, col, patch_size)
            pseudocolor = _build_pseudocolor(patch_cube, rgb_bands)
            _, small_scale, large_scale, _, _ = _run_backbone(model, patch_cube, device)
            small_info = _score_branch_tensor(small_scale)
            large_info = _score_branch_tensor(large_scale)
            input_score = _score_input_patch(pseudocolor)
            total_score = float(input_score + small_info["total_score"] + large_info["total_score"])

            if total_score > best_info["total_score"]:
                best_row = int(row)
                best_col = int(col)
                best_info = {
                    "input_score": float(input_score),
                    "small_score": float(small_info["total_score"]),
                    "large_score": float(large_info["total_score"]),
                    "total_score": float(total_score),
                }

    return best_row, best_col, best_info


def _add_flow_arrow(fig: plt.Figure, ax_from: plt.Axes, ax_to: plt.Axes) -> None:
    start = ax_from.get_position()
    end = ax_to.get_position()
    arrow = FancyArrowPatch(
        (start.x1, (start.y0 + start.y1) * 0.5),
        (end.x0, (end.y0 + end.y1) * 0.5),
        transform=fig.transFigure,
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.4,
        color="#4d4d4d",
    )
    fig.add_artist(arrow)


def _plot_map(ax: plt.Axes, image: np.ndarray, title: str, cmap: str = "turbo") -> None:
    if image.ndim == 3:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save_metadata(
    output_path: str,
    image_path: str,
    center_row: int,
    center_col: int,
    patch_size: int,
    rgb_bands: List[int],
    large_branch: List[Dict[str, object]],
    small_branch: List[Dict[str, object]],
    fused_branch: Dict[str, object],
    fusion_source: str,
    search_mode: str,
    search_stride: int,
    search_info: Dict[str, float],
    font_name: str,
) -> None:
    info = {
        "image_path": os.path.abspath(image_path),
        "center_row": int(center_row),
        "center_col": int(center_col),
        "patch_size": int(patch_size),
        "rgb_bands_zero_based": [int(v) for v in rgb_bands],
        "rgb_bands_one_based": [int(v) + 1 for v in rgb_bands],
        "fusion_source": fusion_source,
        "search_mode": search_mode,
        "search_stride": int(search_stride),
        "search_info": search_info,
        "font_name": font_name,
        "large_scale_branch": [
            {
                "label": item["label"],
                "channel_idx": int(item["channel_idx"]),
                "scores": item["scores"],
            }
            for item in large_branch
        ],
        "small_scale_branch": [
            {
                "label": item["label"],
                "channel_idx": int(item["channel_idx"]),
                "scores": item["scores"],
            }
            for item in small_branch
        ],
        "fused_feature": {
            "label": fused_branch["label"],
            "channel_idx": int(fused_branch["channel_idx"]),
            "scores": fused_branch["scores"],
        },
    }
    meta_path = os.path.splitext(output_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def generate_figure(args: argparse.Namespace) -> str:
    config = Config.from_yaml(args.config) if os.path.exists(args.config) else Config()
    font_name = _configure_chinese_font()
    device = _get_device(
        use_cuda=config.get("device.use_cuda", True),
        gpu_id=args.gpu if args.gpu is not None else config.get("device.gpu_id", 0),
    )

    image_cube, _ = _load_raster_array(args.image, is_label=False)
    image_cube = np.asarray(image_cube, dtype=np.float32)

    band_keep_indices = config.get("data.band_keep_indices", None)
    if band_keep_indices is not None:
        image_cube = image_cube[:, :, np.asarray(band_keep_indices, dtype=np.int64)]

    patch_size = args.patch_size if args.patch_size is not None else int(config.get("model.spatial_size", 13))
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size 必须为正奇数")
    num_bands = int(image_cube.shape[2])
    rgb_bands = _parse_rgb_bands(args.rgb_bands, num_bands)

    model = AG_S2CNN(
        num_bands=num_bands,
        spatial_size=patch_size,
        num_classes=int(config.get("model.num_classes", 1)),
    ).to(device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_ckpt = os.path.join(config.get("data.output_dir", "./outputs"), "checkpoints", "best_model.pth")
        if not os.path.exists(default_ckpt):
            raise FileNotFoundError("未提供 --checkpoint，且默认 best_model.pth 不存在")
        checkpoint_path = default_ckpt

    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    auto_search_patch = bool(args.auto_search_patch or args.center_row is None or args.center_col is None)
    search_stride = int(args.search_stride if args.search_stride is not None else patch_size)

    if auto_search_patch:
        center_row, center_col, search_info = _find_representative_patch(
            image_cube=image_cube,
            model=model,
            device=device,
            patch_size=patch_size,
            rgb_bands=rgb_bands,
            search_stride=search_stride,
        )
        search_mode = "auto"
    else:
        center_row = int(args.center_row)
        center_col = int(args.center_col)
        search_info = {}
        search_mode = "manual"

    patch_cube = _extract_patch(image_cube, center_row, center_col, patch_size)
    _, small_scale, large_scale, concatenated, mapped = _run_backbone(model, patch_cube, device)

    fusion_source = args.fusion_source
    fused_source_tensor = concatenated if fusion_source == "concat" else mapped

    large_branch_maps = _select_representative_channels(
        large_scale,
        labels=["Edge-sensitive", "Fine-texture", "Local-anomaly"],
    )
    small_branch_maps = _select_representative_channels(
        small_scale,
        labels=["Edge-sensitive", "Fine-texture", "Local-anomaly"],
    )
    fused_map = _select_fused_channel(fused_source_tensor)
    pseudocolor = _build_pseudocolor(patch_cube, rgb_bands)

    output_path = args.output
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(18.5, 8.8), dpi=args.dpi)
    gs = GridSpec(
        2,
        5,
        figure=fig,
        width_ratios=[1.25, 1.0, 1.0, 1.0, 1.25],
        wspace=0.12,
        hspace=0.12,
    )

    ax_input = fig.add_subplot(gs[:, 0])
    ax_large_1 = fig.add_subplot(gs[0, 1])
    ax_large_2 = fig.add_subplot(gs[0, 2])
    ax_large_3 = fig.add_subplot(gs[0, 3])
    ax_small_1 = fig.add_subplot(gs[1, 1])
    ax_small_2 = fig.add_subplot(gs[1, 2])
    ax_small_3 = fig.add_subplot(gs[1, 3])
    ax_fused = fig.add_subplot(gs[:, 4])

    _plot_map(
        ax_input,
        pseudocolor,
        "原始伪彩色卫星影像",
    )
    _plot_map(
        ax_large_1,
        large_branch_maps[0]["feature_map"],
        f"大尺度分支 | 边界响应\n通道 {large_branch_maps[0]['channel_idx']}",
    )
    _plot_map(
        ax_large_2,
        large_branch_maps[1]["feature_map"],
        f"大尺度分支 | 纹理响应\n通道 {large_branch_maps[1]['channel_idx']}",
    )
    _plot_map(
        ax_large_3,
        large_branch_maps[2]["feature_map"],
        f"大尺度分支 | 局部异常响应\n通道 {large_branch_maps[2]['channel_idx']}",
    )
    _plot_map(
        ax_small_1,
        small_branch_maps[0]["feature_map"],
        f"小尺度分支 | 边界响应\n通道 {small_branch_maps[0]['channel_idx']}",
    )
    _plot_map(
        ax_small_2,
        small_branch_maps[1]["feature_map"],
        f"小尺度分支 | 纹理响应\n通道 {small_branch_maps[1]['channel_idx']}",
    )
    _plot_map(
        ax_small_3,
        small_branch_maps[2]["feature_map"],
        f"小尺度分支 | 局部异常响应\n通道 {small_branch_maps[2]['channel_idx']}",
    )
    _plot_map(
        ax_fused,
        fused_map["feature_map"],
        f"拼接融合特征激活图\n通道 {fused_map['channel_idx']}",
    )

    fig.text(0.16, 0.91, "(a) 输入影像", ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.49, 0.91, "(b) 多尺度并行分支激活响应", ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.86, 0.91, "(c) 拼接融合结果", ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.42, 0.86, "大尺度分支", ha="center", va="bottom", fontsize=12, fontweight="bold")
    fig.text(0.42, 0.43, "小尺度分支", ha="center", va="bottom", fontsize=12, fontweight="bold")
    fig.text(
        0.5,
        0.975,
        "多尺度卫星提取主干的中间执行效果可视化",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    _add_flow_arrow(fig, ax_input, ax_large_1)
    _add_flow_arrow(fig, ax_input, ax_small_1)
    _add_flow_arrow(fig, ax_large_3, ax_fused)
    _add_flow_arrow(fig, ax_small_3, ax_fused)

    fig.text(
        0.265,
        0.69,
        "多尺度卷积并行提取",
        ha="center",
        va="center",
        fontsize=11,
        color="#4d4d4d",
    )
    fig.text(
        0.265,
        0.28,
        "多尺度卷积并行提取",
        ha="center",
        va="center",
        fontsize=11,
        color="#4d4d4d",
    )
    fig.text(
        0.81,
        0.5,
        "通道拼接融合",
        ha="center",
        va="center",
        fontsize=11,
        color="#4d4d4d",
    )
    fig.text(
        0.5,
        0.035,
        "图注：左侧为自动搜索得到的代表性伪彩色卫星影像 patch；中部为多尺度主干两个并行分支中最具代表性的边界、纹理与局部异常响应；右侧为两个分支特征在通道维拼接后的代表性融合激活图，用于展示多尺度信息互补后的综合表征能力。",
        ha="center",
        va="bottom",
        fontsize=11,
        wrap=True,
    )
    fig.text(
        0.5,
        0.01,
        f"自动搜索中心坐标：({center_row}, {center_col})    patch 尺寸：{patch_size}    中文字体：{font_name}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#4d4d4d",
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    _save_metadata(
        output_path=output_path,
        image_path=args.image,
        center_row=center_row,
        center_col=center_col,
        patch_size=patch_size,
        rgb_bands=rgb_bands,
        large_branch=large_branch_maps,
        small_branch=small_branch_maps,
        fused_branch=fused_map,
        fusion_source=fusion_source,
        search_mode=search_mode,
        search_stride=search_stride,
        search_info=search_info,
        font_name=font_name,
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a paper-ready figure for the multi-scale satellite backbone."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, required=True, help="Path to the hyperspectral image.")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path.")
    parser.add_argument("--center-row", type=int, default=None, help="Patch center row.")
    parser.add_argument("--center-col", type=int, default=None, help="Patch center col.")
    parser.add_argument("--patch-size", type=int, default=None, help="Patch size, default from config.")
    parser.add_argument("--auto-search-patch", action="store_true", help="Automatically search a representative patch.")
    parser.add_argument("--search-stride", type=int, default=None, help="Stride for full-image patch search.")
    parser.add_argument(
        "--rgb-bands",
        type=int,
        nargs=3,
        default=None,
        help="Three band indices. If all are within [1, num_bands], they are treated as 1-based.",
    )
    parser.add_argument(
        "--fusion-source",
        type=str,
        choices=["concat", "mapped"],
        default="concat",
        help="Use concatenated features or channel-mapped features for the fused panel.",
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_path = generate_figure(args)
    print(f"Figure saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
