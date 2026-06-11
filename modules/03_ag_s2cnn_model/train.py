"""
训练脚本

实现 AG-S²CNN 模型的训练流水线，包括训练循环、验证、早停机制等。

需求: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from models.ag_s2cnn import AG_S2CNN
from utils.metrics import MetricsCalculator
from utils.dataset import HyperspectralDataset, load_hyperspectral_data, save_geotiff
from utils.visualization import TrainingVisualizer
from config import Config


def _compute_binary_enhanced_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    hnm_ratio: float,
    hnm_weight: float,
    separation_weight: float,
    separation_margin: float
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = logits.view(-1)
    targets = targets.float().view(-1)
    base_loss = criterion(logits, targets)
    probs = torch.sigmoid(logits)
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    hnm_loss = torch.tensor(0.0, device=logits.device)
    separation_loss = torch.tensor(0.0, device=logits.device)
    if neg_mask.any():
        neg_probs = probs[neg_mask]
        if pos_mask.any():
            k = max(1, int(pos_mask.sum().item() * hnm_ratio))
        else:
            k = max(1, int(neg_probs.numel() * 0.1))
        k = min(k, int(neg_probs.numel()))
        hard_neg_probs, hard_idx_local = torch.topk(neg_probs, k=k, largest=True)
        neg_indices = torch.where(neg_mask)[0]
        hard_indices = neg_indices[hard_idx_local]
        hnm_loss = F.binary_cross_entropy_with_logits(logits[hard_indices], targets[hard_indices], reduction='mean')
        if pos_mask.any():
            pos_mean = probs[pos_mask].mean()
            hard_neg_mean = hard_neg_probs.mean()
            separation_loss = F.relu(separation_margin - (pos_mean - hard_neg_mean))
    total_loss = base_loss + hnm_weight * hnm_loss + separation_weight * separation_loss
    stats = {
        'base_loss': float(base_loss.detach().cpu().item()),
        'hnm_loss': float(hnm_loss.detach().cpu().item()),
        'sep_loss': float(separation_loss.detach().cpu().item())
    }
    return total_loss, stats


def _binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.35
) -> torch.Tensor:
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    mod = (1.0 - p_t).pow(gamma)
    return (alpha_t * mod * bce).mean()


def _binary_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    tp = (probs * targets).sum()
    fp = (probs * (1.0 - targets)).sum()
    fn = ((1.0 - probs) * targets).sum()
    tversky = (tp + 1e-8) / (tp + alpha * fp + beta * fn + 1e-8)
    return 1.0 - tversky


def _apply_tta_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'flip_h':
        return torch.flip(x, dims=[-1])
    if mode == 'flip_v':
        return torch.flip(x, dims=[-2])
    if mode == 'flip_hv':
        return torch.flip(x, dims=[-1, -2])
    return x


def _forward_with_tta(
    model: nn.Module,
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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_config: Optional[Dict[str, float]] = None,
    boost: bool = False,
    boost_cfg: Optional[Dict[str, float]] = None,
    ema_state: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[float, Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    """
    训练一个 epoch
    
    参数:
        model: 模型
        dataloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备 (CPU/GPU)
        epoch: 当前 epoch 数
    
    返回:
        avg_loss: 平均损失
        metrics: 训练指标字典
    
    需求: 8.1, 8.5, 15.2
    """
    model.train()
    if loss_config is None:
        loss_config = {}
    if boost_cfg is None:
        boost_cfg = {}
    hnm_ratio = float(loss_config.get('hnm_ratio', 1.0))
    hnm_weight = float(loss_config.get('hnm_weight', 0.5))
    separation_weight = float(loss_config.get('separation_weight', 0.5))
    separation_margin = float(loss_config.get('separation_margin', 0.1))
    ema_decay = float(boost_cfg.get('ema_decay', 0.999))
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    # 初始化 AMP scaler（仅在 CUDA 可用时启用）
    use_cuda_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_cuda_amp)
    
    try:
        for batch_idx, (x_sat, x_ref, targets) in enumerate(pbar):
            try:
                # 将数据移动到设备
                x_sat = x_sat.to(device)
                x_ref = x_ref.to(device)
                targets = targets.float().to(device)

                if boost:
                    targets = targets.view(-1)
                    mixup_alpha = float(boost_cfg.get('mixup_alpha', 0.2))
                    mixup_prob = float(boost_cfg.get('mixup_prob', 0.6))
                    if mixup_alpha > 0 and np.random.rand() < mixup_prob:
                        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                        perm = torch.randperm(x_sat.size(0), device=x_sat.device)
                        x_sat = lam * x_sat + (1.0 - lam) * x_sat[perm]
                        x_ref = lam * x_ref + (1.0 - lam) * x_ref[perm]
                        targets = lam * targets + (1.0 - lam) * targets[perm]

                    cutmix_prob = float(boost_cfg.get('cutmix_prob', 0.2))
                    if cutmix_prob > 0 and np.random.rand() < cutmix_prob and x_sat.size(0) > 1:
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
                        targets = lam_adj * targets + (1.0 - lam_adj) * targets[perm]
                
                # 前向传播 (使用 AMP)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', enabled=use_cuda_amp):
                    outputs = model(x_sat, x_ref)
                    
                    # 计算损失
                    if outputs.size(1) == 1:
                        loss, _ = _compute_binary_enhanced_loss(
                            logits=outputs,
                            targets=targets,
                            criterion=criterion,
                            hnm_ratio=hnm_ratio,
                            hnm_weight=hnm_weight,
                            separation_weight=separation_weight,
                            separation_margin=separation_margin
                        )
                        if boost:
                            logits_vec = outputs.view(-1)
                            targets_vec = targets.float().view(-1)
                            focal_loss = _binary_focal_loss(
                                logits=logits_vec,
                                targets=targets_vec,
                                gamma=float(boost_cfg.get('focal_gamma', 2.0)),
                                alpha=float(boost_cfg.get('focal_alpha', 0.35))
                            )
                            tversky_loss = _binary_tversky_loss(
                                logits=logits_vec,
                                targets=targets_vec,
                                alpha=float(boost_cfg.get('tversky_alpha', 0.3)),
                                beta=float(boost_cfg.get('tversky_beta', 0.7))
                            )
                            loss = (
                                loss
                                + float(boost_cfg.get('focal_weight', 0.35)) * focal_loss
                                + float(boost_cfg.get('tversky_weight', 0.30)) * tversky_loss
                            )
                    else:
                        loss = criterion(outputs, targets)
                    
                    # 附加 L1 正则化
                    if hasattr(model, 'l1_lambda') and model.l1_lambda > 0:
                        l1_reg = torch.tensor(0., device=device)
                        for param in model.parameters():
                            l1_reg += torch.norm(param, 1)
                        loss += model.l1_lambda * l1_reg
                
                # 反向传播
                if use_cuda_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if boost:
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
                
                # 记录损失
                total_loss += loss.item()
                
                # 记录预测结果
                if outputs.size(1) == 1:
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).int().squeeze()
                    
                    pred_numpy = predicted.cpu().numpy()
                    if pred_numpy.ndim == 0:
                        all_predictions.append(int(pred_numpy))
                    else:
                        all_predictions.extend(pred_numpy.tolist())
                else:
                    _, predicted = torch.max(outputs, 1)
                    all_predictions.extend(predicted.cpu().numpy().tolist())
                    
                all_targets.extend(targets.cpu().numpy().tolist())
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except RuntimeError as e:
                # 捕获 CUDA OOM 错误 (需求 15.2)
                if 'out of memory' in str(e).lower():
                    print(f"\n错误: GPU 内存不足!")
                    print(f"当前批量大小: {x_sat.size(0)}")
                    if torch.cuda.is_available():
                        print(f"GPU 内存使用: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
                        print(f"GPU 内存缓存: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
                        # 清理 GPU 缓存
                        torch.cuda.empty_cache()
                    print("建议:")
                    print("  1. 减小批量大小 (batch_size)")
                    print("  2. 减小模型输入尺寸")
                    print("  3. 使用梯度累积")
                    print("  4. 使用混合精度训练 (AMP)")
                    raise RuntimeError(f"GPU 内存不足: {str(e)}") from e
                else:
                    raise
    
    except Exception as e:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    
    # 计算训练指标
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': np.mean(all_predictions == all_targets)
    }
    
    return avg_loss, metrics, ema_state


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics_calculator: MetricsCalculator,
    epoch: int,
    loss_config: Optional[Dict[str, float]] = None,
    threshold: float = 0.5,
    tta: bool = False,
    tta_modes: Optional[Tuple[str, ...]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    验证模型
    """
    model.eval()
    if loss_config is None:
        loss_config = {}
    hnm_ratio = float(loss_config.get('hnm_ratio', 1.0))
    hnm_weight = float(loss_config.get('hnm_weight', 0.5))
    separation_weight = float(loss_config.get('separation_weight', 0.5))
    separation_margin = float(loss_config.get('separation_margin', 0.1))
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probs = []
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    try:
        with torch.no_grad():
            for x_sat, x_ref, targets in pbar:
                try:
                    # 将数据移动到设备
                    x_sat = x_sat.to(device)
                    x_ref = x_ref.to(device)
                    targets = targets.to(device)
                    
                    # 前向传播 (使用 AMP)
                    with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                        outputs = _forward_with_tta(model, x_sat, x_ref, tta=tta, tta_modes=tta_modes)
                        
                        # 计算损失
                        if outputs.size(1) == 1:
                            loss, _ = _compute_binary_enhanced_loss(
                                logits=outputs,
                                targets=targets,
                                criterion=criterion,
                                hnm_ratio=hnm_ratio,
                                hnm_weight=hnm_weight,
                                separation_weight=separation_weight,
                                separation_margin=separation_margin
                            )
                        else:
                            loss = criterion(outputs, targets)
                    
                    # 记录损失
                    total_loss += loss.item()
                    
                    # 记录预测结果
                    if outputs.size(1) == 1:
                        probs = torch.sigmoid(outputs)
                        predicted = (probs > float(threshold)).int().squeeze()
                        
                        # 处理标量张量和一维张量的情况
                        prob_numpy = probs.squeeze().cpu().numpy()
                        if prob_numpy.ndim == 0:
                            all_probs.append(float(prob_numpy))
                            all_predictions.append(int(predicted.cpu().numpy()))
                        else:
                            all_probs.extend(prob_numpy.tolist())
                            all_predictions.extend(predicted.cpu().numpy().tolist())
                    else:
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs, 1)
                        # 多分类时暂不计算二分类专属AUC
                        all_probs.extend(probs[:, 1].cpu().numpy() if probs.size(1) > 1 else probs.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())
                        
                    all_targets.extend(targets.cpu().numpy())
                    
                    # 更新进度条
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except RuntimeError as e:
                    # 捕获 CUDA OOM 错误 (需求 15.2)
                    if 'out of memory' in str(e).lower():
                        print(f"\n错误: GPU 内存不足!")
                        print(f"当前批量大小: {x_sat.size(0)}")
                        if torch.cuda.is_available():
                            print(f"GPU 内存使用: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
                            print(f"GPU 内存缓存: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
                            # 清理 GPU 缓存
                            torch.cuda.empty_cache()
                        print("建议:")
                        print("  1. 减小批量大小 (batch_size)")
                        print("  2. 减小模型输入尺寸")
                        raise RuntimeError(f"GPU 内存不足: {str(e)}") from e
                    else:
                        raise
    
    except Exception as e:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    
    # 计算详细评估指标
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    detailed_metrics = metrics_calculator.calculate_all_metrics(all_targets, all_predictions, y_prob=all_probs)
    
    metrics = {
        'loss': avg_loss,
        'OA': detailed_metrics['OA'],
        'AA': detailed_metrics['AA'],
        'Kappa': detailed_metrics['Kappa'],
        'F1_macro': detailed_metrics['F1_macro']
    }
    if 'AUC' in detailed_metrics:
        metrics['AUC'] = detailed_metrics['AUC']
    
    return avg_loss, metrics


class EarlyStopping:
    """
    早停机制
    
    监控验证集精度，连续 patience 个 epoch 未提升时停止训练。
    
    参数:
        patience (int): 容忍的 epoch 数，默认 20
        min_delta (float): 最小改进量，默认 0.0
        mode (str): 监控模式，'max' 表示越大越好，'min' 表示越小越好
    
    需求: 8.3
    """
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = 'max'):
        """
        初始化早停机制
        
        参数:
            patience: 容忍的 epoch 数
            min_delta: 最小改进量
            mode: 监控模式 ('max' 或 'min')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # 记录触发早停的最佳 epoch
        self.best_epoch = 0
        
        if mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
        elif mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
        else:
            raise ValueError(f"mode 必须为 'max' 或 'min'，当前值: {mode}")
    
    def __call__(self, score: float, epoch: int = 0) -> bool:
        """
        检查是否应该早停
        
        参数:
            score: 当前监控指标的值
            epoch: 当前的 epoch 数
        
        返回:
            bool: 是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.is_better(score, self.best_score):
            # 性能提升，重置计数器
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # 性能未提升，增加计数器
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    config: Config,
    filepath: str
) -> None:
    """
    保存模型检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前 epoch 数
        metrics: 评估指标
        config: 配置对象
        filepath: 保存路径
    
    需求: 8.4, 8.6, 11.1, 11.2, 11.3, 11.6
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.config,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)
    print(f"检查点已保存到: {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    加载模型检查点
    
    参数:
        filepath: 检查点文件路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
    
    返回:
        checkpoint: 检查点字典
    
    需求: 8.4, 8.6, 11.1, 11.2, 11.3, 11.6
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"检查点文件不存在: {os.path.abspath(filepath)}")
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"检查点已加载: {filepath}")
    print(f"Epoch: {checkpoint['epoch']}, Metrics: {checkpoint['metrics']}")
    
    return checkpoint


def get_device(use_cuda: bool = True, gpu_id: int = 0) -> torch.device:
    """
    获取训练设备（CPU/GPU）
    
    参数:
        use_cuda: 是否使用 CUDA
        gpu_id: GPU ID
    
    返回:
        device: torch.device 对象
    
    需求: 8.7, 8.8
    """
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


def _apply_band_subset(
    image_cube: np.ndarray,
    gsrsl: Dict[int, np.ndarray],
    keep_indices: np.ndarray
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
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


def _split_samples_non_overlapping(
    mapped_ground_truth: np.ndarray,
    by_class: Dict[int, np.ndarray],
    train_ratio: float,
    val_ratio: float,
    spatial_size: int,
    seed: int
) -> Tuple[list, list, list]:
    rng = np.random.default_rng(seed)
    h, w = mapped_ground_truth.shape
    half = spatial_size // 2
    allowed_mask = np.ones((h, w), dtype=bool)
    assigned_mask = np.zeros((h, w), dtype=np.uint8)
    train_samples, val_samples = [], []
    for class_idx, coords in by_class.items():
        if len(coords) == 0:
            continue
        n = len(coords)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        perm = rng.permutation(n)
        train_selected = 0
        val_selected = 0
        # 训练集选择
        for idx in perm:
            if train_selected >= n_train:
                break
            i, j = coords[idx]
            if allowed_mask[i, j]:
                train_samples.append((int(i), int(j), int(class_idx)))
                train_selected += 1
                i0 = max(0, i - half)
                i1 = min(h, i + half + 1)
                j0 = max(0, j - half)
                j1 = min(w, j + half + 1)
                allowed_mask[i0:i1, j0:j1] = False
                assigned_mask[i, j] = 1
        # 验证集选择
        for idx in perm:
            if val_selected >= n_val:
                break
            i, j = coords[idx]
            if assigned_mask[i, j] == 0 and allowed_mask[i, j]:
                val_samples.append((int(i), int(j), int(class_idx)))
                val_selected += 1
                i0 = max(0, i - half)
                i1 = min(h, i + half + 1)
                j0 = max(0, j - half)
                j1 = min(w, j + half + 1)
                allowed_mask[i0:i1, j0:j1] = False
                assigned_mask[i, j] = 2
    # 测试集为剩余未分配且仍存在的该类坐标
    test_samples = []
    for class_idx, coords in by_class.items():
        for i, j in coords:
            if assigned_mask[i, j] == 0:
                test_samples.append((int(i), int(j), int(class_idx)))
    return train_samples, val_samples, test_samples


def _get_label_schema(config: Config) -> Dict[str, object]:
    labels = config.get('labels', {})
    background_class_index = int(labels.get('background_class_index', 0))
    return {
        'class_names': ['Background', labels.get('target_class_name', 'Rich Ore Pegmatite')],
        'gt_target_labels': labels.get('gt_target_labels', [1]),
        'gsrsl_target_labels': labels.get('gsrsl_target_labels', [0, 1, 2]),
        'target_class_index': 1,
        'background_class_index': background_class_index
    }


def _remap_ground_truth_labels(ground_truth: np.ndarray, schema: Dict[str, object], num_classes: int) -> np.ndarray:
    # 全部初始化为背景类
    mapped = np.full_like(ground_truth, fill_value=int(schema['background_class_index']), dtype=np.int64)
    target_class_index = int(schema.get('target_class_index', 1))
    # 将富矿标签映射为目标类
    for source_label in schema['gt_target_labels']:
        mapped[ground_truth == int(source_label)] = target_class_index
    
    # 二分类，标签应为 0 和 1
    valid_classes = {0, 1}
    if not set(np.unique(mapped)).issubset(valid_classes):
        raise ValueError(f"重映射后标签超出二分类范围 0~1: {np.unique(mapped)}")
    return mapped


def _normalize_gsrsl(gsrsl_raw) -> Dict[int, np.ndarray]:
    if isinstance(gsrsl_raw, dict):
        gsrsl_dict = gsrsl_raw
    elif hasattr(gsrsl_raw, 'item'):
        gsrsl_dict = gsrsl_raw.item()
    else:
        raise TypeError(f"GSRSL 数据类型不支持: {type(gsrsl_raw)}")
    normalized = {}
    for k, v in gsrsl_dict.items():
        normalized[int(k)] = np.asarray(v, dtype=np.float32)
    return normalized


def _build_target_gsrsl(gsrsl_raw: Dict[int, np.ndarray], schema: Dict[str, object], num_bands: int) -> Dict[int, np.ndarray]:
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
    # 我们只关心目标富矿的光谱作为参考
    merged = {
        target_class_index: _aggregate(schema['gsrsl_target_labels']),
        background_index: np.zeros(num_bands, dtype=np.float32)
    }
    return merged


def _build_stratified_samples(
    mapped_ground_truth: np.ndarray,
    num_classes: int,
    background_class_index: int,
    background_sample_ratio: float,
    seed: int,
    max_samples_per_class: Optional[int] = None
) -> Dict[int, list]:
    rng = np.random.default_rng(seed)
    by_class = {}
    non_background_count = 0
    for class_idx in range(num_classes):
        coords = np.argwhere(mapped_ground_truth == class_idx)
        rng.shuffle(coords)
        if max_samples_per_class is not None and max_samples_per_class > 0 and len(coords) > max_samples_per_class:
            coords = coords[:max_samples_per_class]
        by_class[class_idx] = coords
        if class_idx != background_class_index:
            non_background_count += len(coords)

    if background_class_index in by_class and background_sample_ratio > 0:
        max_bg = int(non_background_count * background_sample_ratio)
        bg_coords = by_class[background_class_index]
        if max_bg > 0 and len(bg_coords) > max_bg:
            by_class[background_class_index] = bg_coords[:max_bg]
    return by_class


def _split_samples(by_class: Dict[int, np.ndarray], train_ratio: float, val_ratio: float) -> Tuple[list, list, list]:
    train_samples, val_samples, test_samples = [], [], []
    for class_idx, coords in by_class.items():
        n = len(coords)
        if n == 0:
            continue
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_train == 0 and n > 0:
            n_train = 1
        if n_test == 0 and n - n_train > 0:
            n_test = 1
            if n_train + n_val + n_test > n:
                if n_val > 0:
                    n_val -= 1
                else:
                    n_train -= 1
        train_coords = coords[:n_train]
        val_coords = coords[n_train:n_train + n_val]
        test_coords = coords[n_train + n_val:n_train + n_val + n_test]
        train_samples.extend([(int(i), int(j), int(class_idx)) for i, j in train_coords])
        val_samples.extend([(int(i), int(j), int(class_idx)) for i, j in val_coords])
        test_samples.extend([(int(i), int(j), int(class_idx)) for i, j in test_coords])
    return train_samples, val_samples, test_samples


def _sample_label_array(samples: list) -> np.ndarray:
    if len(samples) == 0:
        return np.array([], dtype=np.int64)
    return np.array([s[2] for s in samples], dtype=np.int64)


def _print_distribution(name: str, labels: np.ndarray, num_classes: int, class_names: list):
    print(f"\n{name}分布:")
    for class_idx in range(num_classes):
        count = int(np.sum(labels == class_idx))
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
        print(f"  {class_name}: {count}")


def _build_rgb_composite(image_cube: np.ndarray) -> np.ndarray:
    bands = image_cube.shape[2]
    if bands >= 60:
        band_indices = [50, 30, 20]
    else:
        band_indices = [bands * 2 // 3, bands // 2, bands // 3]
    rgb = image_cube[:, :, band_indices].astype(np.float32)
    min_v = rgb.min()
    max_v = rgb.max()
    if max_v > min_v:
        rgb = (rgb - min_v) / (max_v - min_v)
    return rgb


def _build_dense_samples(height: int, width: int, stride: int, label: int = 0) -> list:
    sample_list = []
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            sample_list.append((int(i), int(j), int(label)))
    return sample_list


def _infer_prediction_maps(
    model: nn.Module,
    dataset: HyperspectralDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_shape: Tuple[int, int],
    rich_class_index: int = 0,
    threshold: float = 0.5,
    tta: bool = False,
    tta_modes: Optional[Tuple[str, ...]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    h, w = image_shape
    pred_map = np.full((h, w), -1, dtype=np.int32)
    rich_prob_map = np.zeros((h, w), dtype=np.float32)
    confidence_map = np.zeros((h, w), dtype=np.float32)
    
    # 记录差分或注意力特征激活强度 (作为额外输出)
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
                activations = probs  # 在二分类中，直接用置信度作为激活强度
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

    parser = argparse.ArgumentParser(description='AG-S²CNN 深度学习模型训练脚本')
    parser.add_argument('--config', '-c', type=str, default='config.yaml')
    parser.add_argument('--mode', '-m', type=str, choices=['train', 'validate', 'test'], default='train')
    parser.add_argument('--resume', '-r', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--boost', action='store_true', help='启用 AG-S2CNN 强化训练包（组合损失+Mixup/CutMix+EMA/SWA+CosineLR）')
    parser.add_argument('--boost-ema', type=int, choices=[0, 1], default=None)
    parser.add_argument('--boost-swa', type=int, choices=[0, 1], default=None)
    parser.add_argument('--boost-mixup', type=int, choices=[0, 1], default=None)
    parser.add_argument('--boost-cutmix', type=int, choices=[0, 1], default=None)
    parser.add_argument('--boost-cosine-lr', type=int, choices=[0, 1], default=None)
    parser.add_argument('--tta', action='store_true', help='在验证/测试/密集预测时启用 TTA')
    parser.add_argument('--infer-threshold', type=float, default=None, help='二分类预测阈值，默认沿用配置')

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
    if args.batch_size is not None:
        config.config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config.config['training']['num_epochs'] = args.epochs
    if args.lr is not None:
        config.config['training']['learning_rate'] = args.lr
    if args.output_dir is not None:
        config.config['data']['output_dir'] = args.output_dir
    config.config.setdefault('training', {})
    config.config['training'].setdefault('boost', {})
    if args.boost:
        config.config['training']['boost']['enabled'] = True
    if args.boost_ema is not None:
        config.config['training']['boost']['use_ema'] = bool(args.boost_ema)
    if args.boost_swa is not None:
        config.config['training']['boost']['use_swa'] = bool(args.boost_swa)
    if args.boost_mixup is not None:
        config.config['training']['boost']['use_mixup'] = bool(args.boost_mixup)
    if args.boost_cutmix is not None:
        config.config['training']['boost']['use_cutmix'] = bool(args.boost_cutmix)
    if args.boost_cosine_lr is not None:
        config.config['training']['boost']['use_cosine_lr'] = bool(args.boost_cosine_lr)
    config.config.setdefault('visualization', {})
    if args.tta:
        config.config['visualization']['tta_enabled'] = True
    if args.infer_threshold is not None:
        config.config['visualization']['probability_threshold'] = float(args.infer_threshold)

    device = get_device(
        use_cuda=config.get('device.use_cuda', True),
        gpu_id=config.get('device.gpu_id', 0)
    )

    image_path = config.get('data.image_path', '')
    gt_path = config.get('data.ground_truth_path', '')
    gsrsl_path = config.get('data.gsrsl_path', '')
    output_dir = config.get('data.output_dir', './outputs')
    ensure_directory_exists(output_dir)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    ensure_directory_exists(checkpoint_dir)
    
    validate_file_exists(image_path, "高光谱影像文件")
    validate_file_exists(gt_path, "Ground Truth 标签文件")
    validate_file_exists(gsrsl_path, "GSRSL 标准光谱库文件")

    image_cube, ground_truth_raw, gsrsl_raw, data_metadata = load_hyperspectral_data(
        image_path, gt_path, gsrsl_path, return_metadata=True
    )
    image_cube = np.asarray(image_cube, dtype=np.float32)
    ground_truth_raw = np.asarray(ground_truth_raw)
    if image_cube.ndim != 3:
        raise ValueError(f"影像应为三维数组 (H, W, Bands)，当前形状: {image_cube.shape}")
    if ground_truth_raw.shape != image_cube.shape[:2]:
        raise ValueError(f"标签图尺寸与影像不匹配: {ground_truth_raw.shape} vs {image_cube.shape[:2]}")

    # 可选波段子集选择
    band_keep_indices = config.get('data.band_keep_indices', None)
    if band_keep_indices is not None:
        gsrsl_raw_dict = _normalize_gsrsl(gsrsl_raw)
        image_cube, gsrsl_raw_dict = _apply_band_subset(image_cube, gsrsl_raw_dict, np.asarray(band_keep_indices, dtype=np.int64))
        gsrsl_raw = gsrsl_raw_dict

    inferred_bands = int(image_cube.shape[2])
    config.config['model']['num_bands'] = inferred_bands
    config.config['model']['num_classes'] = 1
    if config.get('model.spatial_size', 13) % 2 == 0:
        raise ValueError("spatial_size 必须为奇数")

    label_schema = _get_label_schema(config)
    class_names = label_schema['class_names']
    target_class_index = int(label_schema.get('target_class_index', 1))
    mapped_ground_truth = _remap_ground_truth_labels(ground_truth_raw, label_schema, num_classes=2)
    gsrsl_dict = _normalize_gsrsl(gsrsl_raw)
    mapped_gsrsl = _build_target_gsrsl(gsrsl_dict, label_schema, inferred_bands)

    background_class_index = int(label_schema['background_class_index'])
    background_sample_ratio = float(config.get('training.background_sample_ratio', 0.5))
    max_samples_per_class = config.get('training.max_samples_per_class', None)
    max_samples_per_class = None if max_samples_per_class in [None, 0] else int(max_samples_per_class)
    by_class = _build_stratified_samples(
        mapped_ground_truth,
        num_classes=2,
        background_class_index=background_class_index,
        background_sample_ratio=background_sample_ratio,
        seed=args.seed,
        max_samples_per_class=max_samples_per_class
    )
    bg_count = len(by_class.get(background_class_index, []))
    target_count = len(by_class.get(target_class_index, []))
    print(f"\n采样后类别统计: Background={bg_count}, Target={target_count}")
    if target_count > 0 and bg_count < target_count:
        print("警告: 采样后背景样本少于目标样本，可能导致全图高亮。建议提高 background_sample_ratio。")

    train_ratio = float(config.get('training.train_ratio', 0.7))
    val_ratio = float(config.get('training.val_ratio', 0.1))
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("训练/验证划分比例非法，需满足 train_ratio > 0, val_ratio >= 0 且 train_ratio + val_ratio < 1")

    spatial_size = int(config.get('model.spatial_size', 13))

    use_non_overlap = bool(config.get('training.non_overlapping_split', True))
    if use_non_overlap:
        train_samples, val_samples, test_samples = _split_samples_non_overlapping(
            mapped_ground_truth=mapped_ground_truth,
            by_class=by_class,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            spatial_size=spatial_size,
            seed=args.seed
        )
        if len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
            train_samples, val_samples, test_samples = _split_samples(by_class, train_ratio, val_ratio)
            if len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
                _print_distribution("总体样本", _sample_label_array([(i,j,c) for c, coords in by_class.items() for i,j in coords]), 4, class_names)
                raise ValueError("非重叠划分与常规划分均得到空集，请检查标签映射与数据分布")
    else:
        train_samples, val_samples, test_samples = _split_samples(by_class, train_ratio, val_ratio)
    if len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
        raise ValueError("划分后训练/验证/测试样本为空，请检查标签映射与数据分布")

    _print_distribution("训练集", _sample_label_array(train_samples), 2, class_names)
    _print_distribution("验证集", _sample_label_array(val_samples), 2, class_names)
    _print_distribution("测试集", _sample_label_array(test_samples), 2, class_names)

    train_dataset = HyperspectralDataset(
        image_cube=image_cube,
        ground_truth=mapped_ground_truth,
        gsrsl=mapped_gsrsl,
        spatial_size=spatial_size,
        augmentation=bool(config.get('augmentation.enabled', True)),
        sample_list=train_samples
    )
    val_dataset = HyperspectralDataset(
        image_cube=image_cube,
        ground_truth=mapped_ground_truth,
        gsrsl=mapped_gsrsl,
        spatial_size=spatial_size,
        augmentation=False,
        sample_list=val_samples
    )
    test_dataset = HyperspectralDataset(
        image_cube=image_cube,
        ground_truth=mapped_ground_truth,
        gsrsl=mapped_gsrsl,
        spatial_size=spatial_size,
        augmentation=False,
        sample_list=test_samples
    )

    batch_size = int(config.get('training.batch_size', 32))
    num_workers = int(config.get('training.num_workers', 0))
    train_labels = _sample_label_array(train_samples)
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    print(f"训练类别计数: class0={int(class_counts[0])}, class1={int(class_counts[1])}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = AG_S2CNN(
        num_bands=inferred_bands,
        spatial_size=spatial_size,
        num_classes=1
    ).to(device)
    model.l1_lambda = float(config.get('training.l1_regularization', 0.006))
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    bg_count_train = max(float(class_counts[0]), 1.0)
    target_count_train = max(float(class_counts[1]), 1.0)
    pos_weight = torch.tensor([bg_count_train / target_count_train], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"损失函数: BCEWithLogitsLoss(pos_weight={float(pos_weight.item()):.4f})")
    loss_config = {
        'hnm_ratio': float(config.get('training.hnm_ratio', 1.0)),
        'hnm_weight': float(config.get('training.hnm_weight', 0.5)),
        'separation_weight': float(config.get('training.separation_weight', 0.5)),
        'separation_margin': float(config.get('training.separation_margin', 0.08))
    }
    print(
        "增强项配置: "
        f"hnm_ratio={loss_config['hnm_ratio']}, "
        f"hnm_weight={loss_config['hnm_weight']}, "
        f"separation_weight={loss_config['separation_weight']}, "
        f"separation_margin={loss_config['separation_margin']}"
    )
    boost_cfg = dict(config.get('training.boost', {}) or {})
    boost_enabled = bool(boost_cfg.get('enabled', False))
    boost_cfg.setdefault('use_ema', True)
    boost_cfg.setdefault('use_swa', True)
    boost_cfg.setdefault('use_mixup', True)
    boost_cfg.setdefault('use_cutmix', True)
    boost_cfg.setdefault('use_cosine_lr', True)
    eval_threshold = float(boost_cfg.get('eval_threshold', 0.5))
    eval_tta = bool(boost_cfg.get('eval_tta', config.get('visualization.tta_enabled', False)))
    raw_tta_modes = boost_cfg.get('tta_modes', None)
    eval_tta_modes = tuple(raw_tta_modes) if isinstance(raw_tta_modes, (list, tuple)) and len(raw_tta_modes) > 0 else None
    if boost_enabled:
        print(
            "Boost配置: "
            f"ema={int(bool(boost_cfg.get('use_ema', True)))}, "
            f"swa={int(bool(boost_cfg.get('use_swa', True)))}, "
            f"mixup={int(bool(boost_cfg.get('use_mixup', True)))}, "
            f"cutmix={int(bool(boost_cfg.get('use_cutmix', True)))}, "
            f"cosine_lr={int(bool(boost_cfg.get('use_cosine_lr', True)))}"
        )

    metrics_calculator = MetricsCalculator(num_classes=1)

    if args.mode in ['validate', 'test']:
        if args.checkpoint is None:
            raise ValueError(f"{args.mode} 模式需要 --checkpoint")
        load_checkpoint(args.checkpoint, model, device=device)
        target_loader = val_loader if args.mode == 'validate' else test_loader
        loss, metrics = validate(
            model,
            target_loader,
            criterion,
            device,
            metrics_calculator,
            epoch=0,
            loss_config=loss_config,
            threshold=eval_threshold,
            tta=eval_tta,
            tta_modes=eval_tta_modes
        )
        model.eval()
        mode_true = []
        mode_pred = []
        with torch.no_grad():
            for x_sat, x_ref, targets in target_loader:
                outputs = _forward_with_tta(
                    model,
                    x_sat.to(device),
                    x_ref.to(device),
                    tta=eval_tta,
                    tta_modes=eval_tta_modes
                )
                if outputs.size(1) == 1:
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > eval_threshold).int().squeeze().cpu().numpy()
                else:
                    predicted = torch.argmax(outputs, dim=1).cpu().numpy()
                mode_pred.extend(predicted.tolist())
                mode_true.extend(targets.numpy().tolist())
        mode_true = np.array(mode_true, dtype=np.int64)
        mode_pred = np.array(mode_pred, dtype=np.int64)
        mode_metrics = metrics_calculator.calculate_all_metrics(mode_true, mode_pred)
        metrics_calculator.print_metrics(mode_metrics, class_names=class_names)
        metrics_calculator.print_confusion_matrix(mode_metrics, class_names=class_names)
        print(f"{args.mode} Loss: {loss:.4f}, OA: {metrics['OA']:.4f}, Kappa: {metrics['Kappa']:.4f}")
        return

    if boost_enabled:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('training.learning_rate', 1e-3),
            weight_decay=config.get('training.weight_decay', 0.016)
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('training.learning_rate', 1e-3),
            weight_decay=config.get('training.weight_decay', 0.016),
            betas=(0.99, 0.999),
            eps=0.001
        )
    scheduler = None
    if boost_enabled:
        if bool(boost_cfg.get('use_cosine_lr', True)):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get('training.num_epochs', 100),
                eta_min=float(config.get('training.learning_rate', 1e-3)) * float(boost_cfg.get('eta_min_ratio', 0.05))
            )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('training.num_epochs', 100),
            eta_min=1e-6
        )
    early_stopping = EarlyStopping(
        patience=config.get('training.early_stopping_patience', 20),
        mode='max'
    )

    start_epoch = 1
    best_val_score = 0.0
    monitor_name = 'AUC/F1_macro'
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch'] + 1
        resume_metrics = checkpoint.get('metrics', {})
        best_val_score = float(resume_metrics.get('AUC', resume_metrics.get('F1_macro', resume_metrics.get('OA', 0.0))))

    num_epochs = int(config.get('training.num_epochs', 100))
    train_losses = []
    val_losses = []
    learning_rates = []
    ema_state: Optional[Dict[str, torch.Tensor]] = None
    last_states = []
    swa_keep = max(2, int(boost_cfg.get('swa_keep', 5)))

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            train_loss, train_metrics, ema_state = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                loss_config=loss_config,
                boost=boost_enabled,
                boost_cfg=boost_cfg,
                ema_state=ema_state
            )
            restore_state = None
            if boost_enabled and bool(boost_cfg.get('use_ema', True)) and ema_state is not None:
                restore_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.load_state_dict({k: v.to(device) for k, v in ema_state.items()})
            val_loss, val_metrics = validate(
                model,
                val_loader,
                criterion,
                device,
                metrics_calculator,
                epoch,
                loss_config=loss_config,
                threshold=eval_threshold,
                tta=eval_tta,
                tta_modes=eval_tta_modes
            )
            if restore_state is not None:
                model.load_state_dict(restore_state)
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step()
            if boost_enabled and bool(boost_cfg.get('use_swa', True)):
                last_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                if len(last_states) > swa_keep:
                    last_states.pop(0)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            learning_rates.append(current_lr)
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss {train_loss:.4f} Acc {train_metrics['accuracy']:.4f} | "
                f"Val Loss {val_loss:.4f} OA {val_metrics['OA']:.4f} Kappa {val_metrics['Kappa']:.4f} "
                f"AUC {val_metrics.get('AUC', 0.0):.4f} F1 {val_metrics.get('F1_macro', 0.0):.4f} | "
                f"LR {current_lr:.6e}"
            )
            current_score = float(val_metrics.get('AUC', val_metrics.get('F1_macro', val_metrics['OA'])))
            if current_score > best_val_score:
                best_val_score = current_score
                save_restore_state = None
                if boost_enabled and bool(boost_cfg.get('use_ema', True)) and ema_state is not None:
                    save_restore_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    model.load_state_dict({k: v.to(device) for k, v in ema_state.items()})
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_metrics,
                    config,
                    os.path.join(checkpoint_dir, 'best_model.pth')
                )
                if save_restore_state is not None:
                    model.load_state_dict(save_restore_state)
            if early_stopping(current_score, epoch):
                print(f"早停触发: 连续 {early_stopping.patience} 个 epoch 无提升. 最佳 epoch 为: {early_stopping.best_epoch}")
                break
    except KeyboardInterrupt:
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch if 'epoch' in locals() else 0,
            {'interrupted': True, 'best_val_score': best_val_score, 'monitor_name': monitor_name},
            config,
            os.path.join(checkpoint_dir, 'interrupted_checkpoint.pth')
        )
        print("训练已中断并保存检查点")
        return

    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model, device=device)
    elif boost_enabled and bool(boost_cfg.get('use_swa', True)) and len(last_states) >= 2:
        avg_state = {}
        for k in last_states[0].keys():
            avg_state[k] = torch.stack([s[k] for s in last_states], dim=0).mean(dim=0)
        model.load_state_dict({k: v.to(device) for k, v in avg_state.items()})

    test_loss, test_metrics = validate(
        model,
        test_loader,
        criterion,
        device,
        metrics_calculator,
        epoch=0,
        loss_config=loss_config,
        threshold=eval_threshold,
        tta=eval_tta,
        tta_modes=eval_tta_modes
    )

    model.eval()
    all_true = []
    all_pred = []
    all_prob = []
    with torch.no_grad():
        for x_sat, x_ref, targets in test_loader:
            outputs = _forward_with_tta(
                model,
                x_sat.to(device),
                x_ref.to(device),
                tta=eval_tta,
                tta_modes=eval_tta_modes
            )
            if outputs.size(1) == 1:
                probs = torch.sigmoid(outputs)
                predicted = (probs > eval_threshold).int().squeeze()
                
                prob_numpy = probs.squeeze().cpu().numpy()
                pred_numpy = predicted.cpu().numpy()
                
                if prob_numpy.ndim == 0:
                    all_prob.append(float(prob_numpy))
                    all_pred.append(int(pred_numpy))
                else:
                    all_prob.extend(prob_numpy.tolist())
                    all_pred.extend(pred_numpy.tolist())
            else:
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                all_prob.extend(probs[:, 1].cpu().numpy().tolist() if probs.size(1) > 1 else probs.cpu().numpy().tolist())
                all_pred.extend(predicted.cpu().numpy().tolist())
                
            all_true.extend(targets.numpy().tolist())
            
    all_true = np.array(all_true, dtype=np.int64)
    all_pred = np.array(all_pred, dtype=np.int64)
    all_prob = np.array(all_prob, dtype=np.float32)
    
    full_test_metrics = metrics_calculator.calculate_all_metrics(all_true, all_pred, y_prob=all_prob)
    metrics_calculator.print_metrics(full_test_metrics, class_names=class_names)
    if 'AUC' in full_test_metrics:
        print(f"AUC: {full_test_metrics['AUC']:.4f}")
    metrics_calculator.print_confusion_matrix(full_test_metrics, class_names=class_names)

    visualizer = TrainingVisualizer(save_dir=output_dir, geo_info=data_metadata.get('image', {}))
    visualizer.plot_loss_curves(train_losses, val_losses)
    visualizer.plot_learning_rate_curve(learning_rates)
    visualizer.plot_confusion_matrix(
        full_test_metrics['confusion_matrix'],
        class_names=class_names,
        save_name='confusion_matrix_test.png',
        title='测试集混淆矩阵'
    )
    visualizer.plot_confusion_matrix(
        full_test_metrics['confusion_matrix'],
        class_names=class_names,
        save_name='confusion_matrix_test_normalized.png',
        title='测试集混淆矩阵（归一化）',
        normalize=True
    )
    visualizer.plot_class_performance(
        precision=full_test_metrics['Precision_per_class'],
        recall=full_test_metrics['Recall_per_class'],
        f1_scores=full_test_metrics['F1_per_class'],
        class_names=class_names,
        save_name='class_performance_test.png',
        title='测试集各类别性能'
    )

    vis_cfg = config.get('visualization', {})
    dense_stride = int(vis_cfg.get('dense_stride', 4))
    dense_batch_size = int(vis_cfg.get('dense_batch_size', batch_size))
    dense_num_workers = int(vis_cfg.get('dense_num_workers', num_workers))
    overlay_threshold = float(vis_cfg.get('probability_threshold', vis_cfg.get('ore_probability_threshold', 0.5)))
    max_dense_points = int(vis_cfg.get('max_dense_points', 120000))
    dense_stride = max(1, dense_stride)
    image_h, image_w = image_cube.shape[:2]
    dense_samples = _build_dense_samples(image_h, image_w, dense_stride, label=0)
    if max_dense_points > 0 and len(dense_samples) > max_dense_points:
        dense_samples = dense_samples[:max_dense_points]

    dense_dataset = HyperspectralDataset(
        image_cube=image_cube,
        ground_truth=mapped_ground_truth,
        gsrsl=mapped_gsrsl,
        spatial_size=spatial_size,
        augmentation=False,
        sample_list=dense_samples
    )
    
    # 强制将已知正样本加入密集预测列表中，防止因为步长 (stride) 被跳过
    known_positives = [(int(i), int(j), target_class_index) for i, j in by_class.get(target_class_index, [])]
    # 将 known_positives 合并进 dense_samples（去重）
    dense_coords = set((s[0], s[1]) for s in dense_samples)
    for kp in known_positives:
        if (kp[0], kp[1]) not in dense_coords:
            dense_samples.append((kp[0], kp[1], 0))
            dense_coords.add((kp[0], kp[1]))
            
    # 更新 dataset 的 samples
    dense_dataset.samples = dense_samples

    dense_pred_map, dense_rich_prob_map, dense_confidence_map, dense_activation_map = _infer_prediction_maps(
        model=model,
        dataset=dense_dataset,
        device=device,
        batch_size=dense_batch_size,
        num_workers=dense_num_workers,
        image_shape=(image_h, image_w),
        rich_class_index=target_class_index,
        threshold=overlay_threshold,
        tta=bool(vis_cfg.get('tta_enabled', eval_tta)),
        tta_modes=tuple(vis_cfg.get('tta_modes')) if isinstance(vis_cfg.get('tta_modes', None), (list, tuple)) and len(vis_cfg.get('tta_modes')) > 0 else eval_tta_modes
    )
    dense_pred_map_filled = dense_pred_map.copy()
    dense_pred_map_filled[dense_pred_map_filled < 0] = int(label_schema['background_class_index'])
    rich_prob_for_visual = _build_clean_visual_map(dense_rich_prob_map, dense_pred_map, dense_stride)
    activation_for_visual = _build_clean_visual_map(dense_activation_map, dense_pred_map, dense_stride)
    
    # 获取单波段灰度图作为底图
    if image_cube.shape[2] > 0:
        gray_image = image_cube[:, :, image_cube.shape[2] // 2]
    else:
        gray_image = np.zeros((image_cube.shape[0], image_cube.shape[1]), dtype=np.float32)
        
    visualizer.plot_prediction_map(
        prediction_map=dense_pred_map_filled,
        class_names=class_names,
        save_name='prediction_map_dense.png',
        title=f'空间预测结果图 (stride={dense_stride})',
        gray_image=gray_image,
        target_class_idx=target_class_index
    )
    visualizer.plot_probability_heatmap(
        probability_map=rich_prob_for_visual,
        save_name='rich_ore_probability_map.png',
        title='富矿置信度热力图'
    )
    
    # 新增: 绘制特征激活热力图
    visualizer.plot_probability_heatmap(
        probability_map=activation_for_visual,
        save_name='feature_activation_map.png',
        title='模型特征激活响应图 (Target Activation)'
    )
    
    visualizer.plot_prediction_overlay(
        gray_image=gray_image,
        probability_map=rich_prob_for_visual,
        threshold=overlay_threshold,
        save_name='ore_potential_overlay.png',
        title='高置信度靶区叠加图',
        overlay_layer_save_name='ore_potential_overlay_layer.png'
    )

    target_class_idx = int(label_schema.get('target_class_index', 1))
    background_class_idx = int(label_schema.get('background_class_index', 0))
    gt_target_mask = mapped_ground_truth == target_class_idx
    gt_background_mask = mapped_ground_truth == background_class_idx
    target_probs = dense_rich_prob_map[gt_target_mask]
    background_probs = dense_rich_prob_map[gt_background_mask]

    if target_probs.size > 0 and background_probs.size > 0:
        visualizer.plot_error_spatial_map(
            prediction_map=dense_pred_map_filled,
            ground_truth=mapped_ground_truth,
            target_class_idx=target_class_idx,
            background_class_idx=background_class_idx,
            class_names=class_names,
            save_name='error_spatial_map.png',
            title='TP/FP/FN/TN 空间错误类型分布图',
            gray_image=gray_image
        )
        visualizer.plot_probability_histogram(
            background_probs=background_probs,
            target_probs=target_probs,
            save_name='gt_rich_vs_background_probability_histogram.png',
            title='GT富矿 vs 背景概率直方图'
        )

        thresholds = np.linspace(0.0, 1.0, 101)
        # 仅在实际完成推理的采样像元上评估阈值曲线，
        # 避免把 stride 未覆盖区域误当作 0 概率负样本。
        sampled_eval_mask = dense_confidence_map > 0
        recalls = []
        precisions = []
        for thr in thresholds:
            pred_pos = (dense_rich_prob_map >= thr) & sampled_eval_mask
            tp = np.logical_and(pred_pos, gt_target_mask & sampled_eval_mask).sum()
            fn = np.logical_and(~pred_pos, gt_target_mask & sampled_eval_mask).sum()
            fp = np.logical_and(pred_pos, gt_background_mask & sampled_eval_mask).sum()
            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recalls.append(recall)
            precisions.append(precision)

        visualizer.plot_threshold_recall_precision_curve(
            thresholds=thresholds,
            recalls=np.asarray(recalls, dtype=np.float32),
            precisions=np.asarray(precisions, dtype=np.float32),
            save_name='threshold_recall_precision_curve.png',
            title='阈值-召回/精确率曲线'
        )

        best_recall_idx = int(np.argmax(recalls))
        best_f1 = 0.0
        best_f1_thr = 0.5
        for thr, r, p in zip(thresholds, recalls, precisions):
            f1 = 2 * p * r / (p + r + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_f1_thr = float(thr)
        print(f"GT概率统计: target_mean={float(target_probs.mean()):.4f}, background_mean={float(background_probs.mean()):.4f}")
        print(f"阈值分析: 最大Recall={float(recalls[best_recall_idx]):.4f}@thr={float(thresholds[best_recall_idx]):.2f}, 最佳F1={best_f1:.4f}@thr={best_f1_thr:.2f}")
    
    # 新增: 保存预测结果到 TIFF 文件
    try:
        from utils.file_utils import save_geotiff
        save_geotiff(
            array=dense_pred_map_filled.astype(np.int32),
            output_path=os.path.join(output_dir, 'prediction_map_dense.tif'),
            reference_tiff_path=image_path
        )
        save_geotiff(
            array=dense_rich_prob_map.astype(np.float32),
            output_path=os.path.join(output_dir, 'rich_ore_probability_map.tif'),
            reference_tiff_path=image_path
        )
        print("空间推理结果的 GeoTIFF 文件已保存")
    except ImportError:
        print("保存 GeoTIFF 文件失败: 找不到 save_geotiff 函数或 GDAL。请检查 utils.file_utils。")
    except Exception as e:
        print(f"保存 GeoTIFF 文件失败: {str(e)}")

    visualizer.close_all()

    geotiff_outputs = {}
    image_geo = data_metadata.get('image', {})
    if image_geo.get('georef_available', False):
        prediction_tif = os.path.join(output_dir, 'prediction_map_dense.tif')
        rich_prob_tif = os.path.join(output_dir, 'rich_ore_probability_map.tif')
        confidence_tif = os.path.join(output_dir, 'prediction_confidence_map.tif')
        try:
            from utils.file_utils import save_geotiff
            save_geotiff(
                output_path=prediction_tif,
                array=dense_pred_map_filled.astype(np.int32),
                reference_tiff_path=image_path
            )
            save_geotiff(
                output_path=rich_prob_tif,
                array=dense_rich_prob_map.astype(np.float32),
                reference_tiff_path=image_path
            )
            save_geotiff(
                output_path=confidence_tif,
                array=dense_confidence_map.astype(np.float32),
                reference_tiff_path=image_path
            )
            geotiff_outputs = {
                'prediction_map_tif': prediction_tif,
                'rich_ore_probability_tif': rich_prob_tif,
                'prediction_confidence_tif': confidence_tif
            }
        except ImportError:
            print("保存 GeoTIFF 失败：缺少 GDAL。")
        except Exception as e:
            print(f"保存 GeoTIFF 失败：{str(e)}")

    report = {
        'best_val_score': float(best_val_score),
        'monitor_name': monitor_name,
        'test_loss': float(test_loss),
        'test_metrics': {
            'OA': float(full_test_metrics['OA']),
            'AA': float(full_test_metrics['AA']),
            'Kappa': float(full_test_metrics['Kappa']),
            'F1_macro': float(full_test_metrics['F1_macro']),
            'F1_per_class': full_test_metrics['F1_per_class'].tolist(),
            'Precision_per_class': full_test_metrics['Precision_per_class'].tolist(),
            'Recall_per_class': full_test_metrics['Recall_per_class'].tolist(),
            'confusion_matrix': full_test_metrics['confusion_matrix'].tolist()
        },
        'class_names': class_names,
        'label_schema': label_schema,
        'num_bands': inferred_bands,
        'spatial_size': spatial_size,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'dense_visualization': {
            'dense_stride': dense_stride,
            'dense_samples': len(dense_samples),
            'ore_probability_threshold': overlay_threshold,
            'max_dense_points': max_dense_points
        },
        'input_metadata': {
            'image': {
                'format': data_metadata.get('image', {}).get('format'),
                'crs': data_metadata.get('image', {}).get('crs'),
                'transform': data_metadata.get('image', {}).get('transform')
            },
            'ground_truth': {
                'format': data_metadata.get('ground_truth', {}).get('format'),
                'crs': data_metadata.get('ground_truth', {}).get('crs'),
                'transform': data_metadata.get('ground_truth', {}).get('transform')
            },
            'georef_check': data_metadata.get('georef_check', {})
        },
        'geotiff_outputs': geotiff_outputs
    }
    report_path = os.path.join(output_dir, 'metrics_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    config_save_path = os.path.join(output_dir, 'config_used.yaml')
    config.save(config_save_path)
    print(f"\n训练完成，最佳验证 {monitor_name}: {best_val_score:.4f}")
    print(f"测试集 OA: {full_test_metrics['OA']:.4f}, Kappa: {full_test_metrics['Kappa']:.4f}")
    print(f"报告文件: {report_path}")


if __name__ == '__main__':
    main()
