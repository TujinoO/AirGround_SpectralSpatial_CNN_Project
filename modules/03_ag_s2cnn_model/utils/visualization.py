"""
可视化功能模块

实现训练过程的可视化功能，包括损失曲线、学习率曲线和混淆矩阵热力图。

需求: 14.3, 14.4, 14.5
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any, Tuple
from matplotlib.figure import Figure
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.font_manager import FontProperties


class TrainingVisualizer:
    """
    训练可视化器
    
    功能:
        1. 生成损失曲线图（训练损失 vs 验证损失）
        2. 生成学习率变化曲线图
        3. 生成混淆矩阵热力图
    
    参数:
        save_dir (str): 图表保存目录
        dpi (int): 图像分辨率
        style (str): Matplotlib 样式
    
    需求: 14.3, 14.4, 14.5
    """
    
    def __init__(
        self,
        save_dir: str = 'results',
        dpi: int = 300,
        style: str = 'seaborn-v0_8-whitegrid',
        geo_info: Optional[Dict[str, Any]] = None
    ):
        """
        初始化可视化器
        
        参数:
            save_dir: 图表保存目录
            dpi: 图像分辨率
            style: Matplotlib 样式
        """
        self.save_dir = save_dir
        self.dpi = dpi
        self.geo_info = geo_info or {}
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置 Matplotlib 样式
        try:
            plt.style.use(style)
        except:
            # 如果样式不可用，使用默认样式
            plt.style.use('default')
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK SC',
            'WenQuanYi Zen Hei', 'DejaVu Sans', 'Arial Unicode MS'
        ]
        plt.rcParams['axes.unicode_minus'] = False
        self.cn_font = self._resolve_chinese_font()
        self._apply_publication_rcparams()

    def _resolve_chinese_font(self) -> Optional[FontProperties]:
        candidate_paths = [
            r'C:\Windows\Fonts\msyh.ttc',
            r'C:\Windows\Fonts\msyhbd.ttc',
            r'C:\Windows\Fonts\simhei.ttf',
            r'C:\Windows\Fonts\simsun.ttc',
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                return FontProperties(fname=path)
        return None

    def _apply_publication_rcparams(self) -> None:
        plt.rcParams.update({
            'figure.figsize': (7.2, 4.8),
            'figure.dpi': float(self.dpi),
            'savefig.dpi': float(self.dpi),
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
            'axes.titlesize': 16,
            'axes.titleweight': 'semibold',
            'axes.labelsize': 14,
            'axes.labelpad': 6,
            'axes.linewidth': 1.1,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'legend.fontsize': 12,
            'legend.frameon': True,
            'legend.framealpha': 0.92,
            'legend.borderpad': 0.4,
            'legend.handlelength': 2.0,
            'lines.linewidth': 2.4,
            'grid.alpha': 0.22,
            'grid.linestyle': '--',
            'grid.linewidth': 0.8,
        })

    def _get_training_curve_palette(self) -> Dict[str, str]:
        return {
            'train': '#0078FF',
            'val': '#FF4D4F',
            'lr': '#00A86B',
            'grid': '#D9DDE4',
            'spine': '#4F5D75',
            'tick': '#30343F'
        }

    def _style_training_curve_axes(
        self,
        fig: Figure,
        ax,
        title: str,
        ylabel: str
    ) -> None:
        palette = self._get_training_curve_palette()
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.set_xlabel('Epoch', fontsize=13, fontweight='semibold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='semibold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax.grid(True, axis='y', color=palette['grid'], linestyle='--', linewidth=0.9, alpha=0.9)
        ax.grid(False, axis='x')
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=11,
            colors=palette['tick'],
            width=1.1,
            length=4
        )
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for spine in ax.spines.values():
            spine.set_color(palette['spine'])
            spine.set_linewidth(1.15)
        legend = ax.legend(
            loc='best',
            fontsize=11,
            frameon=True,
            facecolor='white',
            edgecolor='#C8CED8',
            framealpha=0.96
        )
        if legend is not None:
            for leg_line in legend.get_lines():
                leg_line.set_linewidth(3.4)

    def _normalize_gray_image(self, gray_image: np.ndarray) -> np.ndarray:
        gray = np.asarray(gray_image, dtype=np.float32)
        if gray.ndim == 3:
            if gray.shape[2] == 3:
                gray = np.mean(gray, axis=2)
            else:
                gray = np.squeeze(gray)
        finite_mask = np.isfinite(gray)
        if not np.any(finite_mask):
            return np.zeros_like(gray, dtype=np.float32)
        finite_values = gray[finite_mask]
        p2, p98 = np.percentile(finite_values, [2.0, 98.0])
        if np.isfinite(p2) and np.isfinite(p98) and p98 > p2:
            gray = np.clip(gray, p2, p98)
            gray = (gray - p2) / (p98 - p2)
        else:
            gray_min = float(np.nanmin(gray))
            gray_max = float(np.nanmax(gray))
            if np.isfinite(gray_min) and np.isfinite(gray_max) and gray_max > gray_min:
                gray = (gray - gray_min) / (gray_max - gray_min)
        gray = np.power(np.clip(gray, 0.0, 1.0), 0.92)
        gray = 0.24 + 0.56 * gray
        return np.nan_to_num(gray, nan=0.24, posinf=0.80, neginf=0.24).astype(np.float32)

    def _dilate_mask(self, mask: np.ndarray, radius: int = 1) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if radius <= 0:
            return mask
        result = mask.copy()
        h, w = mask.shape[:2]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                y_src_start = max(0, -dy)
                y_src_end = min(h, h - dy)
                x_src_start = max(0, -dx)
                x_src_end = min(w, w - dx)
                y_dst_start = max(0, dy)
                y_dst_end = min(h, h + dy)
                x_dst_start = max(0, dx)
                x_dst_end = min(w, w + dx)
                result[y_dst_start:y_dst_end, x_dst_start:x_dst_end] |= mask[y_src_start:y_src_end, x_src_start:x_src_end]
        return result

    def _build_probability_overlay_rgba(
        self,
        probability_map: np.ndarray,
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        prob = np.asarray(probability_map, dtype=np.float32)
        thr = float(threshold)
        target_mask = prob >= thr
        cmap = cm.get_cmap('turbo')
        norm = colors.Normalize(vmin=thr, vmax=1.0)
        rgba = cmap(norm(np.clip(prob, thr, 1.0))).astype(np.float32)
        hsv = colors.rgb_to_hsv(rgba[..., :3])
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.35 + 0.08, 0.0, 1.0)
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.20 + 0.06, 0.0, 1.0)
        rgba[..., :3] = colors.hsv_to_rgb(hsv)
        alpha = (np.clip(prob, thr, 1.0) - thr) / max(1e-6, 1.0 - thr)
        alpha = 0.45 + 0.55 * alpha
        rgba[..., 3] = alpha * target_mask.astype(np.float32)
        coverage = float(target_mask.mean()) if target_mask.size > 0 else 0.0
        return rgba.astype(np.float32), target_mask, coverage

    def _get_geo_extent_and_labels(self, shape_2d: Tuple[int, int]):
        height, width = int(shape_2d[0]), int(shape_2d[1])
        transform_obj = self.geo_info.get('transform_obj', None)
        crs_obj = self.geo_info.get('crs_obj', None)
        crs_text = self.geo_info.get('crs', None)
        if transform_obj is None:
            return None, '列坐标', '行坐标', False

        cols = np.array([0.0, float(width), 0.0, float(width)], dtype=np.float64)
        rows = np.array([0.0, 0.0, float(height), float(height)], dtype=np.float64)
        xs = np.asarray(transform_obj.a * cols + transform_obj.b * rows + transform_obj.c, dtype=np.float64)
        ys = np.asarray(transform_obj.d * cols + transform_obj.e * rows + transform_obj.f, dtype=np.float64)

        if crs_obj is not None or crs_text:
            try:
                import rasterio
                from rasterio.warp import transform as crs_transform
                src_crs = crs_obj if crs_obj is not None else rasterio.crs.CRS.from_string(str(crs_text))
                if src_crs is not None and str(src_crs).upper() not in ['EPSG:4326', 'OGC:CRS84']:
                    lon, lat = crs_transform(src_crs, 'EPSG:4326', xs.tolist(), ys.tolist())
                    xs = np.asarray(lon, dtype=np.float64)
                    ys = np.asarray(lat, dtype=np.float64)
                    return [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())], '经度', '纬度', True
                if src_crs is not None and str(src_crs).upper() in ['EPSG:4326', 'OGC:CRS84']:
                    return [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())], '经度', '纬度', True
            except Exception:
                pass

        if crs_text:
            return [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())], f'X ({crs_text})', f'Y ({crs_text})', False
        return [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())], 'X', 'Y', False

    def _to_dms_text(self, value: float, is_lon: bool) -> str:
        v = float(value)
        hemi = 'E' if is_lon else 'N'
        if is_lon and v < 0:
            hemi = 'W'
        if (not is_lon) and v < 0:
            hemi = 'S'
        av = abs(v)
        deg = int(np.floor(av))
        minutes_raw = (av - deg) * 60.0
        minute = int(np.floor(minutes_raw))
        sec = int(np.round((minutes_raw - minute) * 60.0))
        if sec >= 60:
            sec = 0
            minute += 1
        if minute >= 60:
            minute = 0
            deg += 1
        return f"{deg}°{minute:02d}′{sec:02d}″{hemi}"

    def _to_decimal_degree_text(self, value: float, is_lon: bool) -> str:
        v = float(value)
        hemi = 'E' if is_lon else 'N'
        if is_lon and v < 0:
            hemi = 'W'
        if (not is_lon) and v < 0:
            hemi = 'S'
        return f"{abs(v):.5f}°{hemi}"

    def _apply_geo_axis_format(self, ax, xlabel: str, ylabel: str, use_dms: bool):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if not use_dms:
            return
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: self._to_dms_text(x, True)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: self._to_dms_text(y, False)))
        plt.setp(ax.get_xticklabels(), rotation=18, ha='right')

    def _apply_spatial_aspect(
        self,
        ax,
        extent: Optional[List[float]],
        use_dms: bool
    ) -> None:
        if extent is None:
            ax.set_aspect('equal', adjustable='box')
            return
        if use_dms:
            lat_center = 0.5 * (float(extent[2]) + float(extent[3]))
            cos_lat = float(np.cos(np.deg2rad(lat_center)))
            if abs(cos_lat) > 1e-6:
                ax.set_aspect(1.0 / cos_lat, adjustable='box')
                return
        ax.set_aspect('equal', adjustable='box')
    
    def plot_loss_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        save_name: str = 'loss_curves.png',
        title: str = '训练和验证损失曲线'
    ) -> Figure:
        """
        绘制损失曲线图
        
        参数:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            save_name: 保存文件名
            title: 图表标题
        
        返回:
            Figure: Matplotlib 图形对象
        
        需求: 14.3
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        palette = self._get_training_curve_palette()
        
        # 绘制训练损失
        ax.plot(
            epochs,
            train_losses,
            color=palette['train'],
            label='训练损失',
            linewidth=3.2,
            marker='o',
            markersize=5.4,
            markerfacecolor=palette['train'],
            markeredgecolor='white',
            markeredgewidth=0.8,
            solid_capstyle='round'
        )
        
        # 绘制验证损失
        ax.plot(
            epochs,
            val_losses,
            color=palette['val'],
            label='验证损失',
            linewidth=3.2,
            marker='s',
            markersize=5.1,
            markerfacecolor=palette['val'],
            markeredgecolor='white',
            markeredgewidth=0.8,
            solid_capstyle='round'
        )
        
        self._style_training_curve_axes(fig, ax, title, '损失值 (Loss)')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"损失曲线图已保存: {save_path}")
        
        return fig
    
    def plot_learning_rate_curve(
        self,
        learning_rates: List[float],
        save_name: str = 'learning_rate_curve.png',
        title: str = '学习率变化曲线'
    ) -> Figure:
        """
        绘制学习率变化曲线图
        
        参数:
            learning_rates: 学习率列表
            save_name: 保存文件名
            title: 图表标题
        
        返回:
            Figure: Matplotlib 图形对象
        
        需求: 14.4
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(learning_rates) + 1)
        palette = self._get_training_curve_palette()
        
        # 绘制学习率曲线
        ax.plot(
            epochs,
            learning_rates,
            color=palette['lr'],
            label='学习率',
            linewidth=3.4,
            marker='o',
            markersize=5.2,
            markerfacecolor=palette['lr'],
            markeredgecolor='white',
            markeredgewidth=0.8,
            solid_capstyle='round'
        )
        
        self._style_training_curve_axes(fig, ax, title, '学习率 (Learning Rate)')
        
        # 使用科学计数法显示 y 轴
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"学习率曲线图已保存: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_name: str = 'confusion_matrix.png',
        title: str = '混淆矩阵',
        normalize: bool = False,
        cmap: str = 'Blues'
    ) -> Figure:
        """
        绘制混淆矩阵热力图
        
        参数:
            confusion_matrix: 混淆矩阵 (num_classes, num_classes)
            class_names: 类别名称列表
            save_name: 保存文件名
            title: 图表标题
            normalize: 是否归一化（按行归一化为百分比）
            cmap: 颜色映射
        
        返回:
            Figure: Matplotlib 图形对象
        
        需求: 14.5
        """
        # 归一化混淆矩阵
        if normalize:
            cm = confusion_matrix.astype('float')
            row_sums = cm.sum(axis=1, keepdims=True)
            # 避免除以零
            row_sums[row_sums == 0] = 1
            cm = cm / row_sums
            fmt = '.2%'
            vmax = 1.0
        else:
            cm = confusion_matrix
            fmt = 'd'
            vmax = None
        
        # 设置类别名称
        if class_names is None:
            num_classes = confusion_matrix.shape[0]
            class_names = [f'类别 {i}' for i in range(num_classes)]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
        
        # 绘制热力图
        heatmap = sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor='gray',
            vmin=0,
            vmax=vmax,
            ax=ax,
            annot_kws={'fontsize': 12, 'fontweight': 'semibold'},
            cbar_kws={'fraction': 0.046, 'pad': 0.03, 'shrink': 0.95}
        )
        
        # 设置标签和标题
        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
        ax.set_title(title)
        
        # 旋转 x 轴标签
        plt.setp(ax.get_xticklabels(), rotation=18, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0)

        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=11)
        cbar.set_label('比例' if normalize else '样本数', fontsize=12)
        
        # 保存图表
        save_path = os.path.join(self.save_dir, save_name)
        fig.savefig(save_path)
        print(f"混淆矩阵热力图已保存: {save_path}")
        
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, List[float]],
        save_name: str = 'metrics_comparison.png',
        title: str = '评估指标对比'
    ) -> Figure:
        """
        绘制多个评估指标的对比图
        
        参数:
            metrics_dict: 指标字典，格式为 {指标名称: [epoch1_value, epoch2_value, ...]}
            save_name: 保存文件名
            title: 图表标题
        
        返回:
            Figure: Matplotlib 图形对象
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 获取 epoch 数
        first_metric = list(metrics_dict.values())[0]
        epochs = range(1, len(first_metric) + 1)
        
        # 绘制每个指标
        colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'D', 'v', 'p']
        
        for idx, (metric_name, values) in enumerate(metrics_dict.items()):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax.plot(
                epochs,
                values,
                color=color,
                label=metric_name,
                linewidth=2,
                marker=marker,
                markersize=4
            )
        
        # 设置标签和标题
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('指标值', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 添加图例
        ax.legend(loc='best', fontsize=10, ncol=2)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 设置 x 轴为整数
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"指标对比图已保存: {save_path}")
        
        return fig
    
    def plot_class_performance(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        f1_scores: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_name: str = 'class_performance.png',
        title: str = '各类别性能指标'
    ) -> Figure:
        """
        绘制各类别的性能指标柱状图
        
        参数:
            precision: 精确率数组
            recall: 召回率数组
            f1_scores: F1 分数数组
            class_names: 类别名称列表
            save_name: 保存文件名
            title: 图表标题
        
        返回:
            Figure: Matplotlib 图形对象
        """
        # 设置类别名称
        if class_names is None:
            num_classes = len(precision)
            class_names = [f'类别 {i}' for i in range(num_classes)]
        
        precision = np.asarray(precision, dtype=np.float32)
        recall = np.asarray(recall, dtype=np.float32)
        f1_scores = np.asarray(f1_scores, dtype=np.float32)

        # 创建图形
        fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
        
        # 设置柱状图位置
        x = np.arange(len(class_names))
        width = 0.22
        
        # 绘制柱状图
        bars_p = ax.bar(x - width, precision, width, label='Precision', color='#4C78A8', edgecolor='white', linewidth=0.8)
        bars_r = ax.bar(x, recall, width, label='Recall', color='#F58518', edgecolor='white', linewidth=0.8)
        bars_f1 = ax.bar(x + width, f1_scores, width, label='F1-score', color='#54A24B', edgecolor='white', linewidth=0.8)
        
        # 设置标签和标题
        ax.set_xlabel('类别')
        ax.set_ylabel('指标值')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=18, ha='right')
        
        # 添加图例
        ax.legend(loc='upper center', ncol=3)
        
        # 添加网格
        ax.grid(True, axis='y')
        
        # 设置 y 轴范围
        ax.set_ylim([0, 1.05])

        for bars in [bars_p, bars_r, bars_f1]:
            for bar in bars:
                height = float(bar.get_height())
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.015,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )
        
        # 保存图表
        save_path = os.path.join(self.save_dir, save_name)
        fig.savefig(save_path)
        print(f"类别性能图已保存: {save_path}")
        
        return fig

    def plot_prediction_map(
        self,
        prediction_map: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_name: str = 'prediction_map.png',
        title: str = '空间预测结果图',
        gray_image: Optional[np.ndarray] = None,
        target_class_idx: int = 1
    ) -> Figure:
        num_classes = int(np.nanmax(prediction_map)) + 1 if np.isfinite(prediction_map).any() else 4
        if class_names is None:
            class_names = [f'类别 {i}' for i in range(num_classes)]
        class_count = len(class_names)
        if class_count == 2:
            colors_rgba = [
                (0.90, 0.90, 0.90, 1.0),
                (0.84, 0.37, 0.00, 1.0),
            ]
            cmap = ListedColormap(colors_rgba, name='binary_classes')
            norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        else:
            cmap = plt.get_cmap('tab20', class_count)
            norm = None

        extent, xlabel, ylabel, use_dms = self._get_geo_extent_and_labels(prediction_map.shape[:2])
        if gray_image is None:
            fig, ax = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)
            if extent is not None:
                im = ax.imshow(
                    prediction_map,
                    cmap=cmap,
                    norm=norm,
                    vmin=None if norm is not None else 0,
                    vmax=None if norm is not None else class_count - 1,
                    origin='upper',
                    extent=extent,
                    interpolation='nearest'
                )
            else:
                im = ax.imshow(
                    prediction_map,
                    cmap=cmap,
                    norm=norm,
                    vmin=None if norm is not None else 0,
                    vmax=None if norm is not None else class_count - 1,
                    interpolation='nearest'
                )

            if class_count == 2:
                from matplotlib.patches import Patch
                handles = [
                    Patch(facecolor=cmap(0), edgecolor='black', linewidth=0.6, label=class_names[0]),
                    Patch(facecolor=cmap(1), edgecolor='black', linewidth=0.6, label=class_names[1]),
                ]
                ax.legend(handles=handles, loc='upper right')
            else:
                cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
                cbar.set_ticks(np.arange(class_count))
                cbar.set_ticklabels(class_names)

            ax.set_title(title)
            self._apply_geo_axis_format(ax, xlabel, ylabel, use_dms)
            self._apply_spatial_aspect(ax, extent, use_dms)
            save_path = os.path.join(self.save_dir, save_name)
            fig.savefig(save_path)
            print(f"空间预测结果图已保存: {save_path}")
            return fig

        gray = self._normalize_gray_image(gray_image)
        target_idx = int(target_class_idx)
        target_mask = np.asarray(prediction_map == target_idx, dtype=bool)
        display_target_mask = self._dilate_mask(target_mask, radius=1)
        rich_color = colors.to_rgba('#FF2D2D')
        bg_color = colors.to_rgba('#D9D9D9')

        fig = plt.figure(figsize=(12.6, 7.6), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[16, 2.4], width_ratios=[1, 1])
        ax_overlay = fig.add_subplot(gs[0, 0])
        ax_raw = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[1, :])
        ax_legend.axis('off')

        for ax in [ax_overlay, ax_raw]:
            if extent is not None:
                ax.imshow(gray, cmap='gray', origin='upper', extent=extent)
            else:
                ax.imshow(gray, cmap='gray')

        overlay_rgba = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.float32)
        overlay_rgba[display_target_mask, 0] = rich_color[0]
        overlay_rgba[display_target_mask, 1] = rich_color[1]
        overlay_rgba[display_target_mask, 2] = rich_color[2]
        overlay_rgba[display_target_mask, 3] = 0.96
        if extent is not None:
            ax_overlay.imshow(overlay_rgba, origin='upper', extent=extent, interpolation='nearest')
        else:
            ax_overlay.imshow(overlay_rgba, interpolation='nearest')

        display_pred = np.zeros_like(prediction_map, dtype=np.int32)
        display_pred[display_target_mask] = target_idx
        raw_cmap = ListedColormap([bg_color, rich_color], name='prediction_dual')
        raw_norm = BoundaryNorm([-0.5, 0.5, 1.5], raw_cmap.N)
        if extent is not None:
            ax_raw.imshow(display_pred, cmap=raw_cmap, norm=raw_norm, origin='upper', extent=extent, interpolation='nearest')
        else:
            ax_raw.imshow(display_pred, cmap=raw_cmap, norm=raw_norm, interpolation='nearest')

        ax_overlay.set_title('灰度叠加图')
        ax_raw.set_title('预测分类图')
        self._apply_geo_axis_format(ax_overlay, xlabel, ylabel, use_dms)
        self._apply_geo_axis_format(ax_raw, xlabel, ylabel, use_dms)
        self._apply_spatial_aspect(ax_overlay, extent, use_dms)
        self._apply_spatial_aspect(ax_raw, extent, use_dms)

        from matplotlib.patches import Patch
        rich_label = class_names[target_idx] if 0 <= target_idx < len(class_names) else '富矿样本'
        bg_label = class_names[0] if len(class_names) > 0 else '背景'
        handles = [
            Patch(facecolor=rich_color, edgecolor='black', linewidth=0.8, label=f'{rich_label}（左图仅显示该类）'),
            Patch(facecolor=bg_color, edgecolor='black', linewidth=0.8, label=f'{bg_label}（仅右图显示）'),
        ]
        legend_kwargs = {}
        if self.cn_font is not None:
            legend_kwargs['prop'] = self.cn_font.copy()
            legend_kwargs['prop'].set_size(13)
        ax_legend.legend(handles=handles, loc='center', ncol=2, frameon=False, handlelength=1.8, columnspacing=2.2, **legend_kwargs)
        text_kwargs = {}
        if self.cn_font is not None:
            text_kwargs['fontproperties'] = self.cn_font.copy()
            text_kwargs['fontproperties'].set_size(15)
        ax_legend.text(0.5, 0.05, title, ha='center', va='bottom', fontweight='semibold', **text_kwargs)

        save_path = os.path.join(self.save_dir, save_name)
        fig.savefig(save_path)
        print(f"空间预测结果图已保存: {save_path}")
        return fig

    def plot_probability_heatmap(
        self,
        probability_map: np.ndarray,
        save_name: str = 'ore_probability_heatmap.png',
        title: str = '富矿置信度热力图'
    ) -> Figure:
        prob = np.asarray(probability_map, dtype=np.float32)
        fig, ax = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)
        extent, xlabel, ylabel, use_dms = self._get_geo_extent_and_labels(probability_map.shape[:2])
        if extent is not None:
            im = ax.imshow(prob, cmap='turbo', vmin=0, vmax=1, origin='upper', extent=extent)
        else:
            im = ax.imshow(prob, cmap='turbo', vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.95)
        cbar.set_label('富矿概率', fontsize=12)
        cbar.ax.tick_params(labelsize=11)
        ax.set_title(title)
        self._apply_geo_axis_format(ax, xlabel, ylabel, use_dms)
        self._apply_spatial_aspect(ax, extent, use_dms)
        finite_prob = prob[np.isfinite(prob)]
        if finite_prob.size > 0:
            q10, q50, q90 = np.quantile(finite_prob, [0.1, 0.5, 0.9])
            ax.text(
                0.985, 0.02,
                f"q10={q10:.3f}  q50={q50:.3f}  q90={q90:.3f}",
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.75', alpha=0.9)
            )
        save_path = os.path.join(self.save_dir, save_name)
        fig.savefig(save_path)
        print(f"富矿置信度热力图已保存: {save_path}")
        return fig

    def plot_prediction_overlay(
        self,
        gray_image: np.ndarray,
        probability_map: np.ndarray,
        threshold: float = 0.6,
        save_name: str = 'target_overlay_map.png',
        title: str = '高置信度靶区叠加图',
        overlay_layer_save_name: Optional[str] = None
    ) -> Figure:
        """
        绘制灰度底图+高亮靶区叠加图
        
        参数:
            gray_image: 原始影像单波段灰度图 (H, W) 或 (H, W, 1)
            probability_map: 预测的富矿概率图 (H, W)
            threshold: 判定为靶区的置信度阈值
        """
        gray = np.asarray(gray_image, dtype=np.float32)
        if gray.ndim == 3:
            if gray.shape[2] == 3:
                gray = np.mean(gray, axis=2)
            else:
                gray = np.squeeze(gray)
                
        gray_min = gray.min()
        gray_max = gray.max()
        if gray_max > gray_min:
            gray = (gray - gray_min) / (gray_max - gray_min)

        prob = np.asarray(probability_map, dtype=np.float32)
        rgba, target_mask, coverage = self._build_probability_overlay_rgba(prob, float(threshold))
        fig, ax = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)
        extent, xlabel, ylabel, use_dms = self._get_geo_extent_and_labels(gray.shape[:2])

        if extent is not None:
            ax.imshow(gray, cmap='gray', origin='upper', extent=extent)
        else:
            ax.imshow(gray, cmap='gray')

        cmap = cm.get_cmap('turbo')
        norm = colors.Normalize(vmin=float(threshold), vmax=1.0)

        if extent is not None:
            ax.imshow(rgba, origin='upper', extent=extent)
        else:
            ax.imshow(rgba)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03, shrink=0.95)
        cbar.set_label('高置信度富矿概率', fontsize=12)
        cbar.ax.tick_params(labelsize=11)
        ax.set_title(title)
        self._apply_geo_axis_format(ax, xlabel, ylabel, use_dms)
        self._apply_spatial_aspect(ax, extent, use_dms)
        ax.text(
            0.985, 0.02,
            f"threshold={float(threshold):.2f}  coverage={coverage:.2%}",
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.75', alpha=0.9)
        )
        save_path = os.path.join(self.save_dir, save_name)
        fig.savefig(save_path)
        print(f"高置信度靶区叠加图已保存: {save_path}")
        if overlay_layer_save_name:
            overlay_layer_path = os.path.join(self.save_dir, overlay_layer_save_name)
            plt.imsave(overlay_layer_path, rgba)
            print(f"透明背景叠加层已保存: {overlay_layer_path}")
        return fig

    def plot_probability_histogram(
        self,
        background_probs: np.ndarray,
        target_probs: np.ndarray,
        save_name: str = 'gt_rich_vs_background_probability_histogram.png',
        title: str = 'GT富矿 vs 背景概率直方图'
    ) -> Figure:
        bg = np.asarray(background_probs, dtype=np.float32).reshape(-1)
        tg = np.asarray(target_probs, dtype=np.float32).reshape(-1)
        bg = bg[np.isfinite(bg)]
        tg = tg[np.isfinite(tg)]
        bins = np.linspace(0.0, 1.0, 51)
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        ax.hist(bg, bins=bins, alpha=0.30, label='真实背景区', color='#4C78A8', density=True, edgecolor='none')
        ax.hist(tg, bins=bins, alpha=0.30, label='真实富矿区', color='#F58518', density=True, edgecolor='none')
        ax.hist(bg, bins=bins, histtype='step', color='#2F4B7C', density=True, linewidth=2.2)
        ax.hist(tg, bins=bins, histtype='step', color='#B14D0B', density=True, linewidth=2.2)
        if bg.size > 0:
            ax.axvline(float(bg.mean()), color='#2F4B7C', linewidth=1.6, linestyle=':')
        if tg.size > 0:
            ax.axvline(float(tg.mean()), color='#B14D0B', linewidth=1.6, linestyle=':')

        ax.set_xlabel('预测为富矿的概率')
        ax.set_ylabel('概率密度')
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.grid(True, axis='y')
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(0.015, 0.995),
            ncol=1,
            borderaxespad=0.3
        )
        info = (
            f"背景像元数: {bg.size}\n"
            f"背景平均富矿概率: {float(bg.mean()):.3f}\n"
            f"富矿像元数: {tg.size}\n"
            f"富矿平均富矿概率: {float(tg.mean()):.3f}\n"
            f"虚线表示各类别的平均预测概率"
        )
        ax.text(
            0.985, 0.83, info,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.75', alpha=0.9)
        )
        save_path = os.path.join(self.save_dir, save_name)
        fig.savefig(save_path)
        print(f"概率分布直方图已保存: {save_path}")
        return fig

    def plot_threshold_recall_precision_curve(
        self,
        thresholds: np.ndarray,
        recalls: np.ndarray,
        precisions: np.ndarray,
        save_name: str = 'threshold_recall_precision_curve.png',
        title: str = '阈值-召回/精确率曲线'
    ) -> Figure:
        thr = np.asarray(thresholds, dtype=np.float32).reshape(-1)
        rec = np.asarray(recalls, dtype=np.float32).reshape(-1)
        pre = np.asarray(precisions, dtype=np.float32).reshape(-1)
        valid = np.isfinite(thr) & np.isfinite(rec) & np.isfinite(pre)
        thr = thr[valid]
        rec = np.clip(rec[valid], 0.0, 1.0)
        pre = np.clip(pre[valid], 0.0, 1.0)
        f1 = 2.0 * pre * rec / (pre + rec + 1e-8)
        best_idx = int(np.nanargmax(f1)) if f1.size > 0 else 0
        best_thr = float(thr[best_idx]) if thr.size > 0 else 0.5
        best_f1 = float(f1[best_idx]) if f1.size > 0 else 0.0
        best_p = float(pre[best_idx]) if pre.size > 0 else 0.0
        best_r = float(rec[best_idx]) if rec.size > 0 else 0.0

        fig, ax = plt.subplots(figsize=(9.2, 6.8), constrained_layout=True)
        label_kwargs = {}
        title_kwargs = {'fontweight': 'bold'}
        legend_kwargs = {
            'loc': 'upper right',
            'fontsize': 14,
            'ncol': 2,
            'framealpha': 0.96,
            'borderpad': 0.55,
            'handlelength': 2.4,
            'columnspacing': 1.4
        }
        text_kwargs = {'fontsize': 14}
        if self.cn_font is not None:
            axis_font = self.cn_font.copy()
            axis_font.set_size(17)
            title_font = self.cn_font.copy()
            title_font.set_size(21)
            title_font.set_weight('bold')
            legend_font = self.cn_font.copy()
            legend_font.set_size(14)
            info_font = self.cn_font.copy()
            info_font.set_size(14)
            label_kwargs['fontproperties'] = axis_font
            title_kwargs['fontproperties'] = title_font
            legend_kwargs['prop'] = legend_font
            text_kwargs['fontproperties'] = info_font
        curve_style = {
            'linewidth': 3.1,
            'alpha': 0.98,
            'solid_capstyle': 'round'
        }
        marker_size = 62

        ax.plot(thr, rec, color='#0072B2', label='Recall', **curve_style)
        ax.plot(thr, pre, color='#D55E00', label='Precision', **curve_style)
        ax.plot(thr, f1, color='#009E73', label='F1-score', **curve_style)
        ax.axvline(best_thr, color='0.22', linestyle=':', linewidth=2.2, label=f'最佳阈值 = {best_thr:.2f}')
        ax.scatter([best_thr], [best_r], color='#0072B2', s=marker_size, zorder=4, edgecolors='white', linewidths=0.9)
        ax.scatter([best_thr], [best_p], color='#D55E00', s=marker_size, zorder=4, edgecolors='white', linewidths=0.9)
        ax.scatter([best_thr], [best_f1], color='#009E73', s=marker_size, zorder=4, edgecolors='white', linewidths=0.9)

        ax.set_xlabel('概率阈值', fontsize=17, **label_kwargs)
        ax.set_ylabel('指标值', fontsize=17, **label_kwargs)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.set_title(title, fontsize=21, **title_kwargs)
        ax.tick_params(axis='both', which='major', labelsize=15, width=1.2, length=5)
        ax.grid(True, which='major', axis='both', alpha=0.28, linestyle='--', linewidth=0.9)
        ax.legend(**legend_kwargs)

        ax.text(
            0.985, 0.08,
            f"best F1={best_f1:.3f} @ thr={best_thr:.2f}\nP={best_p:.3f}, R={best_r:.3f}",
            transform=ax.transAxes,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.42', facecolor='white', edgecolor='0.70', alpha=0.95),
            **text_kwargs
        )
        save_path = os.path.join(self.save_dir, save_name)
        fig.savefig(save_path)
        print(f"阈值-召回/精确率曲线已保存: {save_path}")
        return fig

    def plot_error_spatial_map(
        self,
        prediction_map: np.ndarray,
        ground_truth: np.ndarray,
        target_class_idx: int = 1,
        background_class_idx: int = 0,
        class_names: Optional[List[str]] = None,
        save_name: str = 'error_spatial_map.png',
        title: str = '空间错误类型分布 (TP/FP/FN/TN)',
        gray_image: Optional[np.ndarray] = None
    ) -> Figure:
        pred = np.asarray(prediction_map, dtype=np.int32)
        gt = np.asarray(ground_truth, dtype=np.int32)
        if pred.shape[:2] != gt.shape[:2]:
            raise ValueError(f"prediction_map shape {pred.shape} 与 ground_truth shape {gt.shape} 不一致")

        tgt = int(target_class_idx)
        bg = int(background_class_idx)
        valid = (gt == tgt) | (gt == bg)

        tp = valid & (gt == tgt) & (pred == tgt)
        fp = valid & (gt == bg) & (pred == tgt)
        fn = valid & (gt == tgt) & (pred == bg)
        tn = valid & (gt == bg) & (pred == bg)

        code = np.full(gt.shape[:2], -1, dtype=np.int8)
        code[tn] = 0
        code[fp] = 1
        code[fn] = 2
        code[tp] = 3

        color_fp = colors.to_rgba('#FF5A36')
        color_fn = colors.to_rgba('#2D7FF9')
        color_tp = colors.to_rgba('#00C853')
        color_tn = (0.90, 0.90, 0.90, 1.0)
        cmap = ListedColormap([
            (0.0, 0.0, 0.0, 0.0),
            color_tn,
            color_fp,
            color_fn,
            color_tp,
        ], name='tp_fp_fn_tn')
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        extent, xlabel, ylabel, use_dms = self._get_geo_extent_and_labels(code.shape[:2])
        if class_names is None or len(class_names) <= max(tgt, bg):
            bg_name = f'背景({bg})'
            tgt_name = f'目标({tgt})'
        else:
            bg_name = class_names[bg]
            tgt_name = class_names[tgt]

        n_tp = int(tp.sum())
        n_fp = int(fp.sum())
        n_fn = int(fn.sum())
        n_tn = int(tn.sum())
        n_valid = int(valid.sum())
        denom = max(1, n_valid)

        from matplotlib.patches import Patch
        if gray_image is None:
            fig, ax = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)
            if extent is not None:
                ax.imshow(code, cmap=cmap, norm=norm, origin='upper', extent=extent, interpolation='nearest')
            else:
                ax.imshow(code, cmap=cmap, norm=norm, interpolation='nearest')

            handles = [
                Patch(facecolor=color_tn, edgecolor='black', linewidth=0.6, label=f"TN: {bg_name}→{bg_name} (n={n_tn}, {n_tn/denom:.1%})"),
                Patch(facecolor=color_tp, edgecolor='black', linewidth=0.6, label=f"TP: {tgt_name}→{tgt_name} (n={n_tp}, {n_tp/denom:.1%})"),
                Patch(facecolor=color_fp, edgecolor='black', linewidth=0.6, label=f"FP: {bg_name}→{tgt_name} (n={n_fp}, {n_fp/denom:.1%})"),
                Patch(facecolor=color_fn, edgecolor='black', linewidth=0.6, label=f"FN: {tgt_name}→{bg_name} (n={n_fn}, {n_fn/denom:.1%})"),
            ]
            ax.legend(handles=handles, loc='lower center', ncol=1, bbox_to_anchor=(0.5, -0.02))
            ax.set_title(title)
            self._apply_geo_axis_format(ax, xlabel, ylabel, use_dms)
            self._apply_spatial_aspect(ax, extent, use_dms)
            save_path = os.path.join(self.save_dir, save_name)
            fig.savefig(save_path)
            print(f"错误空间分布图已保存: {save_path}")
            return fig

        gray = self._normalize_gray_image(gray_image)
        fig = plt.figure(figsize=(12.8, 7.6), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[16, 2.6], width_ratios=[1, 1])
        ax_overlay = fig.add_subplot(gs[0, 0])
        ax_raw = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[1, :])
        ax_legend.axis('off')

        for ax in [ax_overlay, ax_raw]:
            if extent is not None:
                ax.imshow(gray, cmap='gray', origin='upper', extent=extent)
            else:
                ax.imshow(gray, cmap='gray')

        overlay_rgba = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.float32)
        overlay_rgba[fp, :3] = color_fp[:3]
        overlay_rgba[fn, :3] = color_fn[:3]
        overlay_rgba[tp, :3] = color_tp[:3]
        overlay_rgba[fp | fn | tp, 3] = 0.94
        if extent is not None:
            ax_overlay.imshow(overlay_rgba, origin='upper', extent=extent, interpolation='nearest')
            ax_raw.imshow(code, cmap=cmap, norm=norm, origin='upper', extent=extent, interpolation='nearest')
        else:
            ax_overlay.imshow(overlay_rgba, interpolation='nearest')
            ax_raw.imshow(code, cmap=cmap, norm=norm, interpolation='nearest')

        ax_overlay.set_title('灰度叠加图')
        ax_raw.set_title('错误类型图')
        self._apply_geo_axis_format(ax_overlay, xlabel, ylabel, use_dms)
        self._apply_geo_axis_format(ax_raw, xlabel, ylabel, use_dms)
        self._apply_spatial_aspect(ax_overlay, extent, use_dms)
        self._apply_spatial_aspect(ax_raw, extent, use_dms)

        handles = [
            Patch(facecolor=color_tp, edgecolor='black', linewidth=0.8, label=f"TP 真正例 (n={n_tp}, {n_tp/denom:.1%})"),
            Patch(facecolor=color_fp, edgecolor='black', linewidth=0.8, label=f"FP 假正例 (n={n_fp}, {n_fp/denom:.1%})"),
            Patch(facecolor=color_fn, edgecolor='black', linewidth=0.8, label=f"FN 假负例 (n={n_fn}, {n_fn/denom:.1%})"),
            Patch(facecolor=color_tn, edgecolor='black', linewidth=0.8, label=f"TN 真负例 (n={n_tn}, {n_tn/denom:.1%}，仅右图显示)"),
        ]
        ax_legend.legend(handles=handles, loc='center', ncol=4, frameon=False, fontsize=12, handlelength=1.8, columnspacing=1.8)
        ax_legend.text(0.5, 0.04, title, ha='center', va='bottom', fontsize=15, fontweight='semibold')

        save_path = os.path.join(self.save_dir, save_name)
        fig.savefig(save_path)
        print(f"错误空间分布图已保存: {save_path}")
        return fig
    
    def close_all(self):
        """关闭所有打开的图形"""
        plt.close('all')


def create_visualizer(
    save_dir: str = 'results',
    dpi: int = 300,
    style: str = 'seaborn-v0_8-whitegrid',
    geo_info: Optional[Dict[str, Any]] = None
) -> TrainingVisualizer:
    """
    创建训练可视化器的便捷函数
    
    参数:
        save_dir: 图表保存目录
        dpi: 图像分辨率
        style: Matplotlib 样式
    
    返回:
        TrainingVisualizer: 可视化器实例
    """
    return TrainingVisualizer(
        save_dir=save_dir,
        dpi=dpi,
        style=style,
        geo_info=geo_info
    )
