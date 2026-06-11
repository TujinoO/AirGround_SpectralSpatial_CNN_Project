import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from config import Config
from generate_multiscale_backbone_figure import (
    _build_pseudocolor,
    _configure_chinese_font,
    _extract_patch,
    _get_device,
    _normalize_to_unit,
    _parse_rgb_bands,
    _project_channel_to_map,
    _score_feature_map,
)
from models.ag_s2cnn import AG_S2CNN
from predict import _build_target_gsrsl, _get_label_schema, _normalize_gsrsl
from train import load_checkpoint
from utils.dataset import _load_raster_array


def _subset_gsrsl(gsrsl: Dict[int, np.ndarray], keep_indices: np.ndarray) -> Dict[int, np.ndarray]:
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    subset = {}
    for key, value in gsrsl.items():
        vec = np.asarray(value, dtype=np.float32)
        subset[int(key)] = vec[keep_indices]
    return subset


def _build_reference_tensor(config: Config, num_bands: int, device: torch.device) -> torch.Tensor:
    gsrsl_path = config.get("data.gsrsl_path", "")
    if not gsrsl_path or not os.path.exists(gsrsl_path):
        raise FileNotFoundError("缺少可用的 GSRSL 光谱库文件，无法生成图 3")

    gsrsl_raw = np.load(gsrsl_path, allow_pickle=True)
    gsrsl_dict = _normalize_gsrsl(gsrsl_raw)

    band_keep_indices = config.get("data.band_keep_indices", None)
    if band_keep_indices is not None:
        gsrsl_dict = _subset_gsrsl(gsrsl_dict, np.asarray(band_keep_indices, dtype=np.int64))

    schema = _get_label_schema(config)
    target_gsrsl = _build_target_gsrsl(gsrsl_dict, schema, num_bands)
    target_idx = int(schema.get("target_class_index", 1))
    ref_spectrum = np.asarray(target_gsrsl[target_idx], dtype=np.float32)
    return torch.from_numpy(ref_spectrum).float().view(1, 1, -1, 1, 1).to(device)


def _run_fusion_pipeline(
    model: AG_S2CNN,
    patch_cube: np.ndarray,
    x_ref: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    patch_tensor = torch.from_numpy(np.transpose(patch_cube, (2, 0, 1))).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        f_sat = model.s_backbone(patch_tensor)
        ref_reduced = model.g_encoder.conv1(x_ref)
        ref_reduced = model.g_encoder.bn1(ref_reduced)
        ref_reduced = model.g_encoder.relu1(ref_reduced)
        ref_encoded = model.g_encoder.conv2(ref_reduced)
        ref_encoded = model.g_encoder.bn2(ref_encoded)
        ref_encoded = model.g_encoder.relu2(ref_encoded)
        f_ref = model.g_encoder._spatial_broadcast(ref_encoded)
        f_diff = torch.abs(f_sat - f_ref)
        f_combined = torch.cat([f_sat, f_diff], dim=1)
        f_fused_pre = model.ag_fusion.fusion_conv(f_combined)
        attention_weight = model.ag_fusion.attention(f_fused_pre)
        f_fused_post = f_fused_pre * attention_weight
    return {
        "f_sat": f_sat,
        "ref_encoded": ref_encoded,
        "f_ref": f_ref,
        "f_diff": f_diff,
        "f_combined": f_combined,
        "f_fused_pre": f_fused_pre,
        "attention_weight": attention_weight,
        "f_fused_post": f_fused_post,
    }


def _score_tensor_channel(feature_tensor: torch.Tensor, channel_idx: int) -> Dict[str, float]:
    feature_np = feature_tensor.detach().cpu().numpy()[0, channel_idx]
    feature_map = _project_channel_to_map(feature_np)
    scores = _score_feature_map(feature_map)
    return {"feature_map": feature_map, "scores": scores}


def _select_fusion_channel(feature_tensor: torch.Tensor) -> Dict[str, object]:
    feature_np = feature_tensor.detach().cpu().numpy()[0]
    best = None
    for channel_idx in range(feature_np.shape[0]):
        feature_map = _project_channel_to_map(feature_np[channel_idx])
        scores = _score_feature_map(feature_map)
        item = {
            "channel_idx": int(channel_idx),
            "feature_map": feature_map,
            "scores": scores,
        }
        if best is None or item["scores"]["total"] > best["scores"]["total"]:
            best = item
    return best


def _overlay_heatmap_on_rgb(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    cmap = plt.get_cmap("turbo")
    heat_rgb = cmap(heatmap)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * rgb + alpha * heat_rgb
    return np.clip(overlay, 0.0, 1.0)


def _aggregate_feature_map(feature_tensor: torch.Tensor) -> np.ndarray:
    feature_np = feature_tensor.detach().cpu().numpy()[0]
    return _normalize_to_unit(np.maximum(feature_np, 0.0).mean(axis=(0, 1)))


def _build_ref_heatmap(ref_encoded: torch.Tensor, max_channels: int = 16) -> np.ndarray:
    ref_np = ref_encoded.detach().cpu().numpy()[0, :, :, 0, 0]
    channel_energy = np.mean(np.maximum(ref_np, 0.0), axis=1)
    order = np.argsort(channel_energy)[::-1]
    keep = order[: min(max_channels, ref_np.shape[0])]
    heatmap = ref_np[keep]
    return _normalize_to_unit(heatmap)


def _search_patch_score(outputs: Dict[str, torch.Tensor], pseudocolor: np.ndarray) -> Dict[str, float]:
    fused_best = _select_fusion_channel(outputs["f_fused_post"])
    diff_map = _aggregate_feature_map(outputs["f_diff"])
    diff_score = _score_feature_map(diff_map)["total"]
    att = outputs["attention_weight"].detach().cpu().numpy().reshape(-1)
    attention_contrast = float(np.max(att) - np.mean(att))
    gray = np.asarray(pseudocolor, dtype=np.float32).mean(axis=2)
    input_score = float(np.std(gray))
    total = float(fused_best["scores"]["total"] + diff_score + attention_contrast + input_score)
    return {
        "input_score": input_score,
        "diff_score": float(diff_score),
        "fused_score": float(fused_best["scores"]["total"]),
        "attention_contrast": attention_contrast,
        "total_score": total,
    }


def _find_representative_patch(
    image_cube: np.ndarray,
    model: AG_S2CNN,
    x_ref: torch.Tensor,
    device: torch.device,
    patch_size: int,
    rgb_bands: List[int],
    search_stride: int,
) -> Tuple[int, int, Dict[str, float]]:
    height, width, _ = image_cube.shape
    half = patch_size // 2
    row_candidates = list(range(half, max(half + 1, height - half), search_stride))
    col_candidates = list(range(half, max(half + 1, width - half), search_stride))
    last_row = height - half - 1
    last_col = width - half - 1
    if last_row >= 0 and row_candidates[-1] != last_row:
        row_candidates.append(last_row)
    if last_col >= 0 and col_candidates[-1] != last_col:
        col_candidates.append(last_col)

    best_row = height // 2
    best_col = width // 2
    best_info = {"total_score": -1.0}

    for row in row_candidates:
        for col in col_candidates:
            patch_cube = _extract_patch(image_cube, row, col, patch_size)
            pseudocolor = _build_pseudocolor(patch_cube, rgb_bands)
            outputs = _run_fusion_pipeline(model, patch_cube, x_ref, device)
            score_info = _search_patch_score(outputs, pseudocolor)
            if score_info["total_score"] > best_info["total_score"]:
                best_row = int(row)
                best_col = int(col)
                best_info = score_info

    return best_row, best_col, best_info


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


def _plot_heatmap(ax: plt.Axes, image: np.ndarray, title: str, xlabel: str, ylabel: str, cmap: str = "magma") -> None:
    ax.imshow(image, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_attention_bar(ax: plt.Axes, attention_weights: np.ndarray, selected_channel: int) -> List[Dict[str, float]]:
    attention_weights = np.asarray(attention_weights, dtype=np.float32).reshape(-1)
    top_indices = np.argsort(attention_weights)[::-1][:10]
    top_values = attention_weights[top_indices]
    colors = ["#d62728" if int(idx) == int(selected_channel) else "#4c72b0" for idx in top_indices]
    ax.bar(range(len(top_indices)), top_values, color=colors, width=0.72)
    ax.set_xticks(range(len(top_indices)))
    ax.set_xticklabels([str(int(idx)) for idx in top_indices], fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("权重", fontsize=10)
    ax.set_xlabel("通道编号", fontsize=10)
    ax.set_title("通道注意力权重 Top-10", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return [{"channel_idx": int(idx), "weight": float(attention_weights[idx])} for idx in top_indices]


def _save_metadata(
    output_path: str,
    image_path: str,
    center_row: int,
    center_col: int,
    patch_size: int,
    rgb_bands: List[int],
    selected_channel: int,
    selected_weight: float,
    top_attention: List[Dict[str, float]],
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
        "selected_channel": int(selected_channel),
        "selected_channel_attention_weight": float(selected_weight),
        "top_attention_channels": top_attention,
        "search_mode": search_mode,
        "search_stride": int(search_stride),
        "search_info": search_info,
        "font_name": font_name,
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

    patch_size = args.patch_size if args.patch_size is not None else int(config.get("model.spatial_size", 25))
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

    x_ref = _build_reference_tensor(config, num_bands, device)

    auto_search_patch = bool(args.auto_search_patch or args.center_row is None or args.center_col is None)
    search_stride = int(args.search_stride if args.search_stride is not None else patch_size * 4)
    if auto_search_patch:
        center_row, center_col, search_info = _find_representative_patch(
            image_cube=image_cube,
            model=model,
            x_ref=x_ref,
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
    pseudocolor = _build_pseudocolor(patch_cube, rgb_bands)
    outputs = _run_fusion_pipeline(model, patch_cube, x_ref, device)
    selected = _select_fusion_channel(outputs["f_fused_post"])
    selected_channel = int(selected["channel_idx"])

    sat_info = _score_tensor_channel(outputs["f_sat"], selected_channel)
    ref_heatmap = _build_ref_heatmap(outputs["ref_encoded"])
    diff_info = _score_tensor_channel(outputs["f_diff"], selected_channel)
    fused_pre_info = _score_tensor_channel(outputs["f_fused_pre"], selected_channel)
    fused_post_info = _score_tensor_channel(outputs["f_fused_post"], selected_channel)
    overlay = _overlay_heatmap_on_rgb(pseudocolor, fused_post_info["feature_map"])

    attention_weights = outputs["attention_weight"].detach().cpu().numpy().reshape(-1)
    selected_weight = float(attention_weights[selected_channel])

    output_path = args.output
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(18.5, 10.2), dpi=args.dpi)
    gs = GridSpec(2, 4, figure=fig, wspace=0.2, hspace=0.28)

    ax_input = fig.add_subplot(gs[0, 0])
    ax_sat = fig.add_subplot(gs[0, 1])
    ax_ref = fig.add_subplot(gs[0, 2])
    ax_diff = fig.add_subplot(gs[0, 3])
    ax_fused_pre = fig.add_subplot(gs[1, 0])
    ax_att = fig.add_subplot(gs[1, 1])
    ax_fused_post = fig.add_subplot(gs[1, 2])
    ax_overlay = fig.add_subplot(gs[1, 3])

    _plot_map(ax_input, pseudocolor, "原始伪彩色卫星影像")
    _plot_map(ax_sat, sat_info["feature_map"], f"卫星特征 $f_{{sat}}$\n通道 {selected_channel}")
    _plot_heatmap(ax_ref, ref_heatmap, "地面先验编码响应图", "降维后光谱深度", "高响应通道")
    _plot_map(ax_diff, diff_info["feature_map"], f"绝对差分 $|f_{{sat}}-f_{{ref}}|$\n通道 {selected_channel}")
    _plot_map(ax_fused_pre, fused_pre_info["feature_map"], f"融合前特征 $Conv([f_{{sat}};f_{{diff}}])$\n通道 {selected_channel}")
    top_attention = _plot_attention_bar(ax_att, attention_weights, selected_channel)
    _plot_map(ax_fused_post, fused_post_info["feature_map"], f"注意力增强后特征\n通道 {selected_channel}")
    _plot_map(ax_overlay, overlay, "增强响应叠加图")

    fig.text(0.14, 0.94, "(a) 输入与双流特征", ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.50, 0.94, "(b) 差分与融合过程", ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.84, 0.94, "(c) 注意力增强结果", ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.98, "注意力增强空地差分融合层的中间执行效果可视化", ha="center", va="top", fontsize=16, fontweight="bold")

    fig.text(
        0.5,
        0.05,
        "图注：上排从左至右依次展示原始伪彩色输入、卫星观测特征、地面先验的编码响应以及二者的绝对差分响应；下排展示差分拼接后的 1×1×1 融合结果、SE 通道注意力权重、注意力增强后的融合特征及其在原始影像上的叠加响应，用于说明空地差分信息如何被显式计算并被选择性增强。由于地面先验在模型中经过空间广播，其空间分布本应一致，因此这里改为展示广播前的编码响应图，以更真实地反映地面先验的内部表征。",
        ha="center",
        va="bottom",
        fontsize=10.8,
        wrap=True,
    )
    fig.text(
        0.5,
        0.018,
        f"自动搜索中心坐标：({center_row}, {center_col})    patch 尺寸：{patch_size}    代表通道：{selected_channel}    该通道注意力权重：{selected_weight:.3f}    中文字体：{font_name}",
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
        selected_channel=selected_channel,
        selected_weight=selected_weight,
        top_attention=top_attention,
        search_mode=search_mode,
        search_stride=search_stride,
        search_info=search_info,
        font_name=font_name,
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a paper-ready figure for AG differential fusion.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--center-row", type=int, default=None)
    parser.add_argument("--center-col", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--auto-search-patch", action="store_true")
    parser.add_argument("--search-stride", type=int, default=None)
    parser.add_argument("--rgb-bands", type=int, nargs=3, default=None)
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
