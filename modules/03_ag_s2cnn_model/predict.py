import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple

from models.ag_s2cnn import AG_S2CNN
from utils.dataset import HyperspectralDataset, save_geotiff, _load_raster_array
from config import Config


def _apply_tta_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'flip_h':
        return torch.flip(x, dims=[-1])
    if mode == 'flip_v':
        return torch.flip(x, dims=[-2])
    if mode == 'flip_hv':
        return torch.flip(x, dims=[-1, -2])
    return x


def _forward_with_tta(
    model: torch.nn.Module,
    x_sat: torch.Tensor,
    x_ref: torch.Tensor,
    tta: bool = False,
    tta_modes: Optional[Tuple[str, ...]] = None
) -> torch.Tensor:
    if not tta:
        return model(x_sat, x_ref)
    modes = tta_modes if tta_modes is not None else ('none', 'flip_h', 'flip_v', 'flip_hv')
    logits_all = []
    for mode in modes:
        logits_all.append(model(_apply_tta_mode(x_sat, mode), x_ref))
    return torch.stack(logits_all, dim=0).mean(dim=0)


def get_device(use_cuda: bool = True, gpu_id: int = 0) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"使用设备: {device} ({torch.cuda.get_device_name(gpu_id)})")
    else:
        device = torch.device('cpu')
        if use_cuda and not torch.cuda.is_available():
            print("警告: CUDA 不可用，回退到 CPU")
        else:
            print("使用设备: CPU")
    return device


def _normalize_gsrsl(gsrsl_raw):
    if isinstance(gsrsl_raw, dict):
        gsrsl_dict = gsrsl_raw
    elif hasattr(gsrsl_raw, 'item'):
        gsrsl_dict = gsrsl_raw.item()
    else:
        raise TypeError(f"GSRSL 数据类型不支持: {type(gsrsl_raw)}")
    out = {}
    for k, v in gsrsl_dict.items():
        out[int(k)] = np.asarray(v, dtype=np.float32)
    return out


def _apply_band_subset(image_cube: np.ndarray, gsrsl: dict, keep_indices: np.ndarray):
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    if keep_indices.ndim != 1:
        raise ValueError("band_keep_indices 必须为一维整数索引数组")
    if np.any(keep_indices < 0) or np.any(keep_indices >= image_cube.shape[2]):
        raise ValueError("band_keep_indices 含有越界索引")
    image_sub = image_cube[:, :, keep_indices]
    gsrsl_sub = {}
    for k, v in gsrsl.items():
        vec = np.asarray(v, dtype=np.float32)
        if vec.shape[0] != image_cube.shape[2]:
            raise ValueError(f"GSRSL 向量长度与影像波段不一致: {vec.shape[0]} vs {image_cube.shape[2]}")
        gsrsl_sub[int(k)] = vec[keep_indices]
    return image_sub, gsrsl_sub


def _build_dense_samples(height: int, width: int, stride: int, label: int = 0) -> list:
    sample_list = []
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            sample_list.append((int(i), int(j), int(label)))
    return sample_list


def _get_label_schema(config: Config) -> dict:
    labels = config.get('labels', {})
    background_class_index = int(labels.get('background_class_index', 0))
    return {
        'class_names': ['Background', labels.get('target_class_name', 'Rich Ore Pegmatite')],
        'gsrsl_target_labels': labels.get('gsrsl_target_labels', [0, 1, 2]),
        'target_class_index': 1,
        'background_class_index': background_class_index
    }


def _build_target_gsrsl(gsrsl_raw: dict, schema: dict, num_bands: int) -> dict:
    def _aggregate(labels):
        vectors = []
        for label in labels:
            label = int(label)
            if label in gsrsl_raw:
                vector = np.asarray(gsrsl_raw[label], dtype=np.float32)
                if vector.shape[0] != num_bands:
                    raise ValueError(f"GSRSL 波段数与影像不一致: label={label}, {vector.shape[0]} vs {num_bands}")
                vectors.append(vector)
        if len(vectors) == 0:
            return np.zeros(num_bands, dtype=np.float32)
        return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)

    background_index = int(schema['background_class_index'])
    target_class_index = int(schema.get('target_class_index', 1))
    merged = {
        target_class_index: _aggregate(schema['gsrsl_target_labels']),
        background_index: np.zeros(num_bands, dtype=np.float32)
    }
    return merged


def _infer_prediction_maps(model: torch.nn.Module, dataset: HyperspectralDataset, device: torch.device,
                           batch_size: int, num_workers: int, image_shape: tuple, rich_class_index: int = 1,
                           threshold: float = 0.5, tta: bool = False, tta_modes: Optional[Tuple[str, ...]] = None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    h, w = image_shape
    pred_map = np.full((h, w), -1, dtype=np.int32)
    rich_prob_map = np.zeros((h, w), dtype=np.float32)
    confidence_map = np.zeros((h, w), dtype=np.float32)
    activation_map = np.zeros((h, w), dtype=np.float32)
    pointer = 0
    model.eval()
    with torch.no_grad():
        for x_sat, x_ref, _ in tqdm(loader, desc='空间推理'):
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                outputs = _forward_with_tta(
                    model,
                    x_sat.to(device),
                    x_ref.to(device),
                    tta=tta,
                    tta_modes=tta_modes
                )
                
            if outputs.size(1) == 1:
                probs = torch.sigmoid(outputs).cpu().numpy().squeeze(axis=1)
                pred = (probs > float(threshold)).astype(np.int32)
                conf = probs
                rich_prob = probs
                activations = probs
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                pred = np.argmax(probs, axis=1)
                conf = np.max(probs, axis=1)
                rich_prob = probs[:, rich_class_index]
                activations = rich_prob
                
            batch_size_now = x_sat.size(0)
            for idx in range(batch_size_now):
                i, j, _ = dataset.samples[pointer + idx]
                pred_map[i, j] = int(pred[idx])
                rich_prob_map[i, j] = float(rich_prob[idx])
                confidence_map[i, j] = float(conf[idx])
                activation_map[i, j] = float(activations[idx])
            pointer += batch_size_now
    return pred_map, rich_prob_map, confidence_map, activation_map


def _smooth_2d_map(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim != 2:
        return x
    pad = np.pad(x, ((1, 1), (1, 1)), mode='edge')
    out = (
        pad[:-2, :-2] + 2.0 * pad[:-2, 1:-1] + pad[:-2, 2:] +
        2.0 * pad[1:-1, :-2] + 4.0 * pad[1:-1, 1:-1] + 2.0 * pad[1:-1, 2:] +
        pad[2:, :-2] + 2.0 * pad[2:, 1:-1] + pad[2:, 2:]
    ) / 16.0
    return out.astype(np.float32)


def _build_clean_visual_map(value_map: np.ndarray, pred_map: np.ndarray, stride: int) -> np.ndarray:
    base = np.asarray(value_map, dtype=np.float32)
    if base.ndim != 2 or stride <= 1:
        return base
    h, w = base.shape
    row_idx = np.clip(np.round(np.arange(h) / float(stride)).astype(np.int32) * int(stride), 0, h - 1)
    col_idx = np.clip(np.round(np.arange(w) / float(stride)).astype(np.int32) * int(stride), 0, w - 1)
    dense = base[np.ix_(row_idx, col_idx)]
    sampled_mask = np.asarray(pred_map >= 0)
    dense[sampled_mask] = base[sampled_mask]
    dense = _smooth_2d_map(dense)
    return np.clip(dense, 0.0, 1.0).astype(np.float32)


def main(args=None):
    import argparse
    from utils.file_utils import validate_file_exists, ensure_directory_exists

    parser = argparse.ArgumentParser(description='AG-S²CNN 预测脚本')
    parser.add_argument('--config', '-c', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--predict-image', type=str, required=True)
    parser.add_argument('--predict-output-dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--tta', action='store_true', help='推理时启用 TTA')
    parser.add_argument('--prob-threshold', type=float, default=None, help='二分类阈值（默认取配置文件 visualization.probability_threshold）')

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if os.path.exists(args.config):
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    if args.gpu is not None:
        config.config['device']['gpu_id'] = args.gpu

    device = get_device(
        use_cuda=config.get('device.use_cuda', True),
        gpu_id=config.get('device.gpu_id', 0)
    )

    image_path = args.predict_image
    gsrsl_path = config.get('data.gsrsl_path', '')
    output_dir = args.predict_output_dir if args.predict_output_dir is not None else config.get('data.output_dir', './outputs')
    ensure_directory_exists(output_dir)

    validate_file_exists(image_path, "高光谱影像文件")
    validate_file_exists(gsrsl_path, "GSRSL 标准光谱库文件")

    image_cube, image_meta = _load_raster_array(image_path, is_label=False)
    image_cube = np.asarray(image_cube, dtype=np.float32)

    band_keep_indices = config.get('data.band_keep_indices', None)
    if band_keep_indices is not None:
        gsrsl_raw = np.load(gsrsl_path, allow_pickle=True)
        gsrsl_raw_dict = _normalize_gsrsl(gsrsl_raw)
        image_cube, gsrsl_raw_dict = _apply_band_subset(image_cube, gsrsl_raw_dict, np.asarray(band_keep_indices, dtype=np.int64))
        gsrsl_raw = gsrsl_raw_dict
    else:
        gsrsl_raw = np.load(gsrsl_path, allow_pickle=True)

    inferred_bands = int(image_cube.shape[2])
    spatial_size = int(config.get('model.spatial_size', 13))
    num_classes = int(config.get('model.num_classes', 1))
    label_schema = _get_label_schema(config)
    mapped_gsrsl = _build_target_gsrsl(_normalize_gsrsl(gsrsl_raw), label_schema, inferred_bands)

    model = AG_S2CNN(num_bands=inferred_bands, spatial_size=spatial_size, num_classes=num_classes).to(device)

    if args.checkpoint is None:
        default_ckpt = os.path.join(config.get('data.output_dir', './outputs'), 'checkpoints', 'best_model.pth')
        if os.path.exists(default_ckpt):
            args.checkpoint = default_ckpt
        else:
            raise ValueError("需要 --checkpoint")

    from train import load_checkpoint
    load_checkpoint(args.checkpoint, model, device=device)

    vis_cfg = config.get('visualization', {})
    dense_stride = int(vis_cfg.get('dense_stride', 1))
    dense_batch_size = int(vis_cfg.get('dense_batch_size', 64))
    dense_num_workers = int(vis_cfg.get('dense_num_workers', 0))
    infer_threshold = float(args.prob_threshold) if args.prob_threshold is not None else float(vis_cfg.get('probability_threshold', 0.5))
    infer_tta = bool(args.tta or vis_cfg.get('tta_enabled', False))
    infer_tta_modes = tuple(vis_cfg.get('tta_modes')) if isinstance(vis_cfg.get('tta_modes', None), (list, tuple)) and len(vis_cfg.get('tta_modes')) > 0 else None

    image_h, image_w = image_cube.shape[:2]
    dense_samples = _build_dense_samples(image_h, image_w, dense_stride, label=0)
    dummy_gt = np.zeros((image_h, image_w), dtype=np.int32)

    dense_dataset = HyperspectralDataset(
        image_cube=image_cube,
        ground_truth=dummy_gt,
        gsrsl=mapped_gsrsl,
        spatial_size=spatial_size,
        augmentation=False,
        sample_list=dense_samples
    )

    print("开始密集空间推理...")
    dense_pred_map, dense_rich_prob_map, dense_confidence_map, dense_activation_map = _infer_prediction_maps(
        model=model,
        dataset=dense_dataset,
        device=device,
        batch_size=dense_batch_size,
        num_workers=dense_num_workers,
        image_shape=(image_h, image_w),
        rich_class_index=1,
        threshold=infer_threshold,
        tta=infer_tta,
        tta_modes=infer_tta_modes
    )

    dense_pred_map_filled = dense_pred_map.copy()
    dense_pred_map_filled[dense_pred_map_filled < 0] = int(label_schema['background_class_index'])

    # 保存预测结果到 TIFF 文件
    if image_meta.get('georef_available', False):
        prediction_tif = os.path.join(output_dir, 'prediction_map_dense.tif')
        rich_prob_tif = os.path.join(output_dir, 'rich_ore_probability_map.tif')
        confidence_tif = os.path.join(output_dir, 'prediction_confidence_map.tif')
        activation_tif = os.path.join(output_dir, 'feature_activation_map.tif')
        save_geotiff(
            output_path=prediction_tif,
            array=dense_pred_map_filled.astype(np.uint8),
            reference_meta=image_meta,
            dtype='uint8',
            nodata=255
        )
        save_geotiff(
            output_path=rich_prob_tif,
            array=dense_rich_prob_map.astype(np.float32),
            reference_meta=image_meta,
            dtype='float32',
            nodata=np.nan
        )
        save_geotiff(
            output_path=confidence_tif,
            array=dense_confidence_map.astype(np.float32),
            reference_meta=image_meta,
            dtype='float32',
            nodata=np.nan
        )
        save_geotiff(
            output_path=activation_tif,
            array=dense_activation_map.astype(np.float32),
            reference_meta=image_meta,
            dtype='float32',
            nodata=np.nan
        )

    rich_prob_for_visual = _build_clean_visual_map(dense_rich_prob_map, dense_pred_map, dense_stride)
    activation_for_visual = _build_clean_visual_map(dense_activation_map, dense_pred_map, dense_stride)

    try:
        from utils.visualization import TrainingVisualizer
        visualizer = TrainingVisualizer(save_dir=output_dir, geo_info=image_meta)
        class_names = label_schema.get('class_names', ['Target', 'Background'])
        gray_image = np.mean(image_cube, axis=2).astype(np.float32)
        gray_image = (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image) + 1e-8)
        
        visualizer.plot_prediction_map(
            prediction_map=dense_pred_map_filled,
            class_names=class_names,
            save_name='prediction_map_dense.png',
            title=f'空间预测结果图 (stride={dense_stride})'
        )
        visualizer.plot_probability_heatmap(
            probability_map=rich_prob_for_visual,
            save_name='rich_ore_probability_map.png',
            title='富矿置信度热力图'
        )
        visualizer.plot_probability_heatmap(
            probability_map=activation_for_visual,
            save_name='feature_activation_map.png',
            title='模型特征激活响应图 (Target Activation)'
        )
        visualizer.plot_prediction_overlay(
            gray_image=gray_image,
            probability_map=rich_prob_for_visual,
            threshold=infer_threshold,
            save_name='ore_potential_overlay.png',
            title='高置信度靶区叠加图',
            overlay_layer_save_name='ore_potential_overlay_layer.png'
        )
        visualizer.close_all()
    except Exception as e:
        print(f"生成可视化结果图失败: {e}")


if __name__ == '__main__':
    main()
