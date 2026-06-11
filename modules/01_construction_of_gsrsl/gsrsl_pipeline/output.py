"""
Output generator module for GSRSL pipeline.
GSRSL管道的输出生成器模块

This module provides functionality to save the Ground Standard Reference Spectral Library
(GSRSL) to file and generate visualizations of the reference spectra.
该模块提供将地面标准参考光谱库(GSRSL)保存到文件并生成参考光谱可视化的功能。
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from gsrsl_pipeline.data_models import GroundSpectrum
from gsrsl_pipeline.filters import apply_savgol_filter

# Configure logger
# 配置日志记录器
logger = logging.getLogger(__name__)


def _select_representative_indices(
    spectra: List[np.ndarray],
    class_mean: np.ndarray,
    anchor: np.ndarray,
    n_samples: int = 5,
) -> List[int]:
    if len(spectra) <= n_samples:
        return list(range(len(spectra)))

    mean_scale = max(float(np.ptp(class_mean)), 1e-6)
    anchor_scale = max(float(np.ptp(anchor)), 1e-6)
    scores = []
    for idx, spectrum in enumerate(spectra):
        mean_rmse = float(np.sqrt(np.mean((spectrum - class_mean) ** 2)) / mean_scale)
        anchor_rmse = float(np.sqrt(np.mean((spectrum - anchor) ** 2)) / anchor_scale)
        scores.append((0.55 * anchor_rmse + 0.45 * mean_rmse, idx))
    return [idx for _, idx in sorted(scores)[:n_samples]]


def save_gsrsl(class_means: Dict[int, np.ndarray], output_path: str) -> None:
    """
    Save GSRSL dictionary to numpy file.
    将GSRSL字典保存到numpy文件
    
    This function saves the Ground Standard Reference Spectral Library (GSRSL)
    to a numpy file format (.npy). The GSRSL contains mean reference spectra
    for five lithology classes, each with 297 spectral bands matching the
    GF-5 satellite specifications.
    
    该函数将地面标准参考光谱库(GSRSL)保存为numpy文件格式(.npy)。
    GSRSL包含五个岩性类别的平均参考光谱，每个类别有297个光谱波段，
    与GF-5卫星规格匹配。
    
    Args:
        class_means: Dictionary mapping class_id to mean spectrum
                    将class_id映射到平均光谱的字典
                    - Keys: integers 0-4 representing lithology classes
                           整数0-4，表示岩性类别
                    - Values: numpy arrays of shape (297,) with dtype float32
                             形状为(297,)、数据类型为float32的numpy数组
        output_path: Path to save gsrsl.npy file
                    保存gsrsl.npy文件的路径
    
    Raises:
        TypeError: If class_means is not a dictionary
                  如果class_means不是字典
        ValueError: If class_means does not have exactly 5 keys (0-4)
                   如果class_means没有恰好5个键(0-4)
        ValueError: If any array has incorrect shape or dtype
                   如果任何数组的形状或数据类型不正确
        ValueError: If any reflectance value is outside [0.0, 1.0] range (excluding NaN)
                   如果任何反射率值超出[0.0, 1.0]范围(不包括NaN)
        OSError: If output directory cannot be created or file cannot be written
                如果无法创建输出目录或无法写入文件
    
    Example:
        >>> # Create sample GSRSL dictionary
        >>> class_means = {
        ...     0: np.full(297, 0.3, dtype=np.float32),
        ...     1: np.full(297, 0.4, dtype=np.float32),
        ...     2: np.full(297, 0.5, dtype=np.float32),
        ...     3: np.full(297, 0.6, dtype=np.float32),
        ...     4: np.full(297, 0.7, dtype=np.float32)
        ... }
        >>> 
        >>> # Save to file
        >>> save_gsrsl(class_means, 'output/gsrsl.npy')
        >>> 
        >>> # Load and verify
        >>> loaded = np.load('output/gsrsl.npy', allow_pickle=True).item()
        >>> assert loaded.keys() == class_means.keys()
    
    Notes:
        - Creates output directory if it doesn't exist
          如果输出目录不存在则创建
        - Validates all arrays have correct shape (297,) and dtype float32
          验证所有数组具有正确的形状(297,)和数据类型float32
        - Validates all non-NaN reflectance values are in range [0.0, 1.0]
          验证所有非NaN反射率值在范围[0.0, 1.0]内
        - Uses numpy.save with allow_pickle=True for dictionary storage
          使用numpy.save和allow_pickle=True进行字典存储
    
    Lithology Classes:
    岩性类别：
        0: Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)
        1: Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)
        2: Mixed-type Rich Pegmatite (混合型富矿伟晶岩)
        3: Barren Pegmatite (贫矿伟晶岩)
        4: Wall Rock (围岩)
    """
    # Validate input type
    # 验证输入类型
    if not isinstance(class_means, dict):
        raise TypeError(
            f"class_means must be a dictionary, got {type(class_means).__name__}"
        )
    
    # Validate dictionary has exactly 5 keys (0-4)
    # 验证字典恰好有5个键(0-4)
    expected_keys = {0, 1, 2, 3, 4}
    actual_keys = set(class_means.keys())
    
    if actual_keys != expected_keys:
        missing_keys = expected_keys - actual_keys
        extra_keys = actual_keys - expected_keys
        error_msg = "class_means must have exactly 5 keys (0, 1, 2, 3, 4). "
        if missing_keys:
            error_msg += f"Missing keys: {sorted(missing_keys)}. "
        if extra_keys:
            error_msg += f"Extra keys: {sorted(extra_keys)}."
        raise ValueError(error_msg)
    
    # Class names for logging
    # 用于日志记录的类别名称
    class_names = {
        0: "Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)",
        1: "Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)",
        2: "Mixed-type Rich Pegmatite (混合型富矿伟晶岩)",
        3: "Barren Pegmatite (贫矿伟晶岩)",
        4: "Wall Rock (围岩)"
    }
    
    # Validate each array
    # 验证每个数组
    for class_id in range(5):
        spectrum = class_means[class_id]
        
        # Check if numpy array
        # 检查是否为numpy数组
        if not isinstance(spectrum, np.ndarray):
            raise ValueError(
                f"Class {class_id} ({class_names[class_id]}): "
                f"spectrum must be a numpy array, got {type(spectrum).__name__}"
            )
        
        # Check shape
        # 检查形状
        if spectrum.shape != (297,):
            raise ValueError(
                f"Class {class_id} ({class_names[class_id]}): "
                f"spectrum must have shape (297,), got {spectrum.shape}"
            )
        
        # Check dtype
        # 检查数据类型
        if spectrum.dtype != np.float32:
            raise ValueError(
                f"Class {class_id} ({class_names[class_id]}): "
                f"spectrum must have dtype float32, got {spectrum.dtype}"
            )
        
        # Validate reflectance range (excluding NaN)
        # 验证反射率范围(不包括NaN)
        valid_values = spectrum[~np.isnan(spectrum)]
        if len(valid_values) > 0:
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            
            if min_val < 0.0 or max_val > 1.0:
                raise ValueError(
                    f"Class {class_id} ({class_names[class_id]}): "
                    f"reflectance values must be in range [0.0, 1.0] (excluding NaN). "
                    f"Found values in range [{min_val:.6f}, {max_val:.6f}]"
                )
        
        # Log statistics
        # 记录统计信息
        n_nan = np.sum(np.isnan(spectrum))
        if len(valid_values) > 0:
            mean_val = np.mean(valid_values)
            logger.info(
                f"Class {class_id} ({class_names[class_id]}): "
                f"{len(valid_values)} valid bands (mean={mean_val:.4f}), "
                f"{n_nan} NaN bands"
            )
        else:
            logger.warning(
                f"Class {class_id} ({class_names[class_id]}): "
                f"all 297 bands are NaN"
            )
    
    # Create output directory if it doesn't exist
    # 如果输出目录不存在则创建
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create if there's a directory component
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory ensured: {output_dir}")
        except OSError as e:
            raise OSError(
                f"Failed to create output directory '{output_dir}': {e}"
            ) from e
    
    # Save using numpy
    # 使用numpy保存
    try:
        np.save(output_path, class_means, allow_pickle=True)
        logger.info(f"GSRSL successfully saved to {output_path}")
        
        # Log file size
        # 记录文件大小
        file_size = os.path.getsize(output_path)
        logger.info(f"Output file size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        
    except Exception as e:
        raise OSError(
            f"Failed to save GSRSL to '{output_path}': {e}"
        ) from e


def visualize_gsrsl(
    class_means: Dict[int, np.ndarray],
    wavelengths: np.ndarray,
    output_path: str
) -> None:
    """
    Generate visualization of all five reference spectra.
    生成所有五个参考光谱的可视化
    
    This function creates a matplotlib plot showing the mean reference spectra
    for all five lithology classes. The plot displays wavelength on the x-axis
    and reflectance on the y-axis, with each class shown in a distinct color.
    
    该函数创建一个matplotlib图，显示所有五个岩性类别的平均参考光谱。
    图在x轴上显示波长，在y轴上显示反射率，每个类别用不同的颜色显示。
    
    Args:
        class_means: Dictionary mapping class_id to mean spectrum
                    将class_id映射到平均光谱的字典
                    - Keys: integers 0-4 representing lithology classes
                           整数0-4，表示岩性类别
                    - Values: numpy arrays of shape (297,) with dtype float32
                             形状为(297,)、数据类型为float32的numpy数组
        wavelengths: Satellite band center wavelengths
                    卫星波段中心波长
                    - numpy array of shape (297,) with wavelengths in nm
                      形状为(297,)的numpy数组，波长单位为nm
        output_path: Path to save gsrsl_visualization.png
                    保存gsrsl_visualization.png的路径
    
    Raises:
        TypeError: If class_means is not a dictionary
                  如果class_means不是字典
        TypeError: If wavelengths is not a numpy array
                  如果wavelengths不是numpy数组
        ValueError: If class_means does not have exactly 5 keys (0-4)
                   如果class_means没有恰好5个键(0-4)
        ValueError: If wavelengths does not have shape (297,)
                   如果wavelengths的形状不是(297,)
        ValueError: If any spectrum array has incorrect shape
                   如果任何光谱数组的形状不正确
        OSError: If output directory cannot be created or file cannot be written
                如果无法创建输出目录或无法写入文件
    
    Example:
        >>> # Create sample data
        >>> class_means = {
        ...     0: np.full(297, 0.3, dtype=np.float32),
        ...     1: np.full(297, 0.4, dtype=np.float32),
        ...     2: np.full(297, 0.5, dtype=np.float32),
        ...     3: np.full(297, 0.6, dtype=np.float32),
        ...     4: np.full(297, 0.7, dtype=np.float32)
        ... }
        >>> wavelengths = np.linspace(400, 2500, 297)
        >>> 
        >>> # Generate visualization
        >>> visualize_gsrsl(class_means, wavelengths, 'output/gsrsl_visualization.png')
    
    Notes:
        - Creates output directory if it doesn't exist
          如果输出目录不存在则创建
        - Uses distinct colors for each lithology class
          为每个岩性类别使用不同的颜色
        - Includes legend with English and Chinese class names
          包含英文和中文类别名称的图例
        - Saves plot as PNG with 300 DPI resolution
          将图保存为300 DPI分辨率的PNG
        - NaN values are handled gracefully by matplotlib (shown as gaps)
          matplotlib优雅地处理NaN值(显示为间隙)
    
    Lithology Classes:
    岩性类别：
        0: Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)
        1: Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)
        2: Mixed-type Rich Pegmatite (混合型富矿伟晶岩)
        3: Barren Pegmatite (贫矿伟晶岩)
        4: Wall Rock (围岩)
    """
    # Validate class_means type
    # 验证class_means类型
    if not isinstance(class_means, dict):
        raise TypeError(
            f"class_means must be a dictionary, got {type(class_means).__name__}"
        )
    
    # Validate wavelengths type
    # 验证wavelengths类型
    if not isinstance(wavelengths, np.ndarray):
        raise TypeError(
            f"wavelengths must be a numpy array, got {type(wavelengths).__name__}"
        )
    
    # Validate dictionary has exactly 5 keys (0-4)
    # 验证字典恰好有5个键(0-4)
    expected_keys = {0, 1, 2, 3, 4}
    actual_keys = set(class_means.keys())
    
    if actual_keys != expected_keys:
        missing_keys = expected_keys - actual_keys
        extra_keys = actual_keys - expected_keys
        error_msg = "class_means must have exactly 5 keys (0, 1, 2, 3, 4). "
        if missing_keys:
            error_msg += f"Missing keys: {sorted(missing_keys)}. "
        if extra_keys:
            error_msg += f"Extra keys: {sorted(extra_keys)}."
        raise ValueError(error_msg)
    
    # Validate wavelengths shape
    # 验证wavelengths形状
    if wavelengths.shape != (297,):
        raise ValueError(
            f"wavelengths must have shape (297,), got {wavelengths.shape}"
        )
    
    # Validate each spectrum array
    # 验证每个光谱数组
    for class_id in range(5):
        spectrum = class_means[class_id]
        if not isinstance(spectrum, np.ndarray):
            raise ValueError(
                f"Class {class_id}: spectrum must be a numpy array, "
                f"got {type(spectrum).__name__}"
            )
        if spectrum.shape != (297,):
            raise ValueError(
                f"Class {class_id}: spectrum must have shape (297,), "
                f"got {spectrum.shape}"
            )
    
    # Class names (English only, for broad compatibility with default fonts)
    class_names = {
        0: "Class 0: Spodumene-rich Pegmatite",
        1: "Class 1: Lepidolite-rich Pegmatite",
        2: "Class 2: Mixed-type Rich Pegmatite",
        3: "Class 3: Barren Pegmatite",
        4: "Class 4: Wall Rock"
    }
    
    # Create output directory if it doesn't exist
    # 如果输出目录不存在则创建
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create if there's a directory component
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory ensured: {output_dir}")
        except OSError as e:
            raise OSError(
                f"Failed to create output directory '{output_dir}': {e}"
            ) from e
    
    # Create figure
    # 创建图形
    plt.figure(figsize=(14, 8))
    
    # Plot each class with distinct color
    # 用不同颜色绘制每个类别
    colors = ['#D7263D', '#1D4ED8', '#15803D', '#D97706', '#6D28D9']
    
    for class_id in range(5):
        spectrum = class_means[class_id]
        plt.plot(
            wavelengths,
            spectrum,
            color=colors[class_id],
            label=class_names[class_id],
            linewidth=3.6,
            alpha=0.98
        )
    
    # Configure plot
    # 配置图
    plt.xlabel('Wavelength / nm', fontsize=18, fontweight='bold')
    plt.ylabel('Reflectance', fontsize=18, fontweight='bold')
    plt.title(
        'Ground Standard Reference Spectral Library (GSRSL)',
        fontsize=21,
        fontweight='bold',
        pad=16
    )
    
    # Add legend
    plt.legend(
        loc='upper right',
        fontsize=14,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True
    )
    
    # Add grid
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set axis limits
    # 设置坐标轴限制
    plt.xlim(wavelengths[0], wavelengths[-1])
    plt.ylim(0.0, 1.0)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    # Add minor ticks
    # 添加次要刻度
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
    
    # Tight layout to prevent label cutoff
    # 紧凑布局以防止标签被截断
    plt.tight_layout()
    
    # Save figure
    # 保存图形
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Visualization successfully saved to {output_path}")
        
        # Log file size
        # 记录文件大小
        file_size = os.path.getsize(output_path)
        logger.info(f"Visualization file size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        
    except Exception as e:
        raise OSError(
            f"Failed to save visualization to '{output_path}': {e}"
        ) from e
    finally:
        # Close figure to free memory
        # 关闭图形以释放内存
        plt.close()
    
    logger.info("Visualization generation completed successfully")


def generate_additional_visualizations(
    ground_spectra: List[GroundSpectrum],
    sat_wavelengths: np.ndarray,
    class_means: Dict[int, np.ndarray],
    example_payload: Dict[int, Dict[str, Any]],
    output_dir: str
) -> Dict[str, str]:
    if len(ground_spectra) == 0:
        raise ValueError("ground_spectra is empty")

    os.makedirs(output_dir, exist_ok=True)

    class_names = {
        0: "Class 0: Spodumene-rich Pegmatite",
        1: "Class 1: Lepidolite-rich Pegmatite",
        2: "Class 2: Mixed-type Rich Pegmatite",
        3: "Class 3: Barren Pegmatite",
        4: "Class 4: Wall Rock"
    }
    colors = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD']
    sample_colors = ['#E7B7B2', '#F1CF96', '#C7DCA0', '#9FCBBC', '#9FB3D7']
    mean_color = '#C73A32'
    style = {
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
    }

    wavelengths = ground_spectra[0].wavelengths
    by_class = {i: [] for i in range(5)}
    for spectrum in ground_spectra:
        if spectrum.class_id in by_class:
            by_class[spectrum.class_id].append(np.asarray(spectrum.reflectances, dtype=np.float64))

    by_class_smooth = {i: [] for i in range(5)}

    smoothed_class_means = {}
    for class_id in range(5):
        if len(by_class[class_id]) == 0:
            continue
        class_matrix = np.vstack(by_class[class_id])
        class_mean = np.mean(class_matrix, axis=0)
        class_mean = np.clip(class_mean, 0.0, 1.0)
        temp = GroundSpectrum(
            wavelengths=np.asarray(wavelengths, dtype=np.float64),
            reflectances=class_mean.astype(np.float64),
            class_id=class_id,
            filename=f"class_{class_id}_mean",
            is_anomalous=False
        )
        smoothed = apply_savgol_filter(temp)
        smoothed_class_means[class_id] = smoothed.reflectances
        for reflectance in by_class[class_id]:
            sample = GroundSpectrum(
                wavelengths=np.asarray(wavelengths, dtype=np.float64),
                reflectances=np.clip(reflectance, 0.0, 1.0).astype(np.float64),
                class_id=class_id,
                filename=f"class_{class_id}_sample",
                is_anomalous=False
            )
            filtered = apply_savgol_filter(sample)
            by_class_smooth[class_id].append(np.asarray(filtered.reflectances, dtype=np.float64))

    output_paths: Dict[str, str] = {}

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(12, 6))
        for class_id in range(5):
            if class_id not in class_means:
                continue
            ax.plot(
                sat_wavelengths,
                class_means[class_id],
                color=colors[class_id],
                linewidth=1.8,
                label=class_names[class_id]
            )
        ax.set_xlabel("Wavelength / nm")
        ax.set_ylabel("Reflectance")
        ax.set_title("Ground Standard Reference Spectral Library (Paper Style)")
        ax.text(0.01, 0.98, "(a)", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_xlim(float(sat_wavelengths[0]), float(sat_wavelengths[-1]))
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        paper_main_path = os.path.join(output_dir, "gsrsl_visualization_paper.png")
        fig.savefig(paper_main_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        output_paths["gsrsl_visualization_paper"] = paper_main_path

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(12, 6))
        for class_id in range(5):
            if class_id not in smoothed_class_means:
                continue
            ax.plot(
                wavelengths,
                smoothed_class_means[class_id],
                color=colors[class_id],
                linewidth=1.6,
                label=class_names[class_id]
            )
        ax.set_xlabel("Wavelength / nm")
        ax.set_ylabel("Reflectance")
        ax.set_title("Mean Ground Spectra by Class (Smoothed)")
        ax.text(0.01, 0.98, "(a)", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_xlim(float(wavelengths[0]), float(wavelengths[-1]))
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        class_mean_path = os.path.join(output_dir, "real_spectra_class_means.png")
        fig.savefig(class_mean_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        output_paths["real_spectra_class_means"] = class_mean_path

    available_classes = [cid for cid in range(5) if cid in example_payload]
    if len(available_classes) > 0:
        with plt.rc_context(style):
            fig, axes = plt.subplots(len(available_classes), 1, figsize=(12, 2.4 * len(available_classes)), sharex=True)
            if len(available_classes) == 1:
                axes = [axes]
            panel_labels = list("abcdefghijklmnopqrstuvwxyz")
            for idx, class_id in enumerate(available_classes):
                payload = example_payload[class_id]
                wl = np.asarray(payload["wavelengths"], dtype=np.float64)
                raw = np.asarray(payload["reflectances"], dtype=np.float64)
                smooth = np.asarray(payload["filtered_reflectances"], dtype=np.float64)
                filename = str(payload["filename"])
                ax = axes[idx]
                ax.plot(wl, raw, color="0.7", linewidth=0.8, label="Original")
                ax.plot(wl, smooth, color=colors[class_id], linewidth=1.4, label="Smoothed")
                ax.set_ylabel("Reflectance")
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.2, linestyle="--")
                ax.set_title(f"{class_names[class_id]} | Example: {filename}")
                ax.text(0.01, 0.95, f"({panel_labels[idx]})", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
                ax.legend(loc="upper right", frameon=True)
            axes[-1].set_xlabel("Wavelength / nm")
            fig.tight_layout()
            examples_path = os.path.join(output_dir, "real_spectra_examples_smoothed.png")
            fig.savefig(examples_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            output_paths["real_spectra_examples_smoothed"] = examples_path

        with plt.rc_context(
            {
                **style,
                'font.size': 16,
                'axes.titlesize': 18,
                'axes.labelsize': 17,
                'legend.fontsize': 15,
                'xtick.labelsize': 15,
                'ytick.labelsize': 15
            }
        ):
            fig, axes = plt.subplots(5, 1, figsize=(15.5, 22), sharex=True)
            legend_handles = [
                Line2D([0], [0], color=sample_colors[0], linewidth=1.7, label="5 Representative Samples"),
                Line2D([0], [0], color=mean_color, linewidth=3.8, label="Class Mean Spectrum"),
            ]
            for class_id, ax in enumerate(axes):
                if class_id not in smoothed_class_means or class_id not in example_payload:
                    continue
                class_mean = np.asarray(smoothed_class_means[class_id], dtype=np.float64)
                anchor = np.asarray(example_payload[class_id]["filtered_reflectances"], dtype=np.float64)
                selected_indices = _select_representative_indices(
                    by_class_smooth[class_id],
                    class_mean,
                    anchor,
                    n_samples=5,
                )
                offsets = np.linspace(-0.028, 0.028, len(selected_indices))
                stacked_curves = [class_mean]
                for rank, (sample_idx, offset) in enumerate(zip(selected_indices, offsets), start=1):
                    shifted_curve = by_class_smooth[class_id][sample_idx] + offset
                    stacked_curves.append(shifted_curve)
                    ax.plot(
                        wavelengths,
                        shifted_curve,
                        color=sample_colors[(rank - 1) % len(sample_colors)],
                        linewidth=1.55,
                        alpha=0.94
                    )
                ax.plot(wavelengths, class_mean, color=mean_color, linewidth=3.6, alpha=0.98)
                curve_stack = np.vstack(stacked_curves)
                y_min = max(0.0, float(np.min(curve_stack)) - 0.03)
                y_max = min(1.0, float(np.max(curve_stack)) + 0.03)
                ax.set_ylim(y_min, y_max)
                ax.set_ylabel("Reflectance")
                ax.set_title(class_names[class_id], pad=12)
                ax.grid(True, alpha=0.22, linestyle="--")
                ax.legend(handles=legend_handles, loc="upper right", frameon=True)
            axes[-1].set_xlabel("Wavelength / nm")
            axes[-1].set_xlim(float(wavelengths[0]), float(wavelengths[-1]))
            fig.suptitle("Representative Ground Spectra and Class Mean Spectra", fontsize=22, y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.986))
            representative_path = os.path.join(output_dir, "representative_spectra_with_class_means.png")
            fig.savefig(representative_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            output_paths["representative_spectra_with_class_means"] = representative_path

        primary_class = available_classes[0]
        payload = example_payload[primary_class]
        wl = np.asarray(payload["wavelengths"], dtype=np.float64)
        raw = np.asarray(payload["reflectances"], dtype=np.float64)
        smooth = np.asarray(payload["filtered_reflectances"], dtype=np.float64)
        sat_reflectance = np.asarray(payload["resampled_reflectances"], dtype=np.float64)

        with plt.rc_context(style):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ax1.plot(wl, raw, color="0.65", linewidth=0.9, label="Original")
            ax1.plot(wl, smooth, color=colors[primary_class], linewidth=1.5, label="Savitzky-Golay Smoothed")
            ax1.set_xlim(float(wl[0]), float(wl[-1]))
            ax1.set_ylim(0.0, 1.0)
            ax1.set_xlabel("Wavelength / nm")
            ax1.set_ylabel("Reflectance")
            ax1.set_title("Savitzky-Golay Filter: Full Spectrum")
            ax1.text(0.01, 0.95, "(a)", transform=ax1.transAxes, va="top", ha="left", fontweight="bold")
            ax1.grid(True, alpha=0.25, linestyle="--")
            ax1.legend(loc="upper right", frameon=True)

            mask = (wl >= 2000.0) & (wl <= 2400.0)
            ax2.plot(wl[mask], raw[mask], color="0.65", linewidth=0.9, label="Original")
            ax2.plot(wl[mask], smooth[mask], color=colors[primary_class], linewidth=1.5, label="Smoothed")
            ax2.set_xlabel("Wavelength / nm")
            ax2.set_ylabel("Reflectance")
            ax2.set_title("Zoom: Absorption Feature near 2200 nm")
            ax2.text(0.01, 0.95, "(b)", transform=ax2.transAxes, va="top", ha="left", fontweight="bold")
            ax2.grid(True, alpha=0.25, linestyle="--")
            ax2.legend(loc="upper left", frameon=True)
            fig.tight_layout()
            savgol_path = os.path.join(output_dir, "savgol_filter_demo.png")
            fig.savefig(savgol_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            output_paths["savgol_filter_demo"] = savgol_path

        with plt.rc_context(style):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ax1.plot(wl, smooth, color=colors[primary_class], linewidth=1.4, label="Ground Spectrum (Smoothed)")
            ax1.set_xlabel("Wavelength / nm")
            ax1.set_ylabel("Reflectance")
            ax1.set_title("High-Resolution Ground Spectrum")
            ax1.text(0.01, 0.95, "(a)", transform=ax1.transAxes, va="top", ha="left", fontweight="bold")
            ax1.grid(True, alpha=0.25, linestyle="--")
            ax1.set_xlim(float(wl[0]), float(wl[-1]))
            ax1.set_ylim(0.0, 1.0)
            ax1.legend(loc="upper right", frameon=True)

            ax2.plot(wl, smooth, color="0.75", linewidth=0.8, label="Ground Spectrum (Smoothed)")
            ax2.plot(sat_wavelengths, sat_reflectance, "o-", color=colors[primary_class], markersize=2.8, linewidth=1.0, label="Resampled to Satellite Bands")
            ax2.set_xlabel("Wavelength / nm")
            ax2.set_ylabel("Reflectance")
            ax2.set_title("Comparison: Ground VS Resampled")
            ax2.text(0.01, 0.95, "(b)", transform=ax2.transAxes, va="top", ha="left", fontweight="bold")
            ax2.grid(True, alpha=0.25, linestyle="--")
            ax2.set_xlim(float(wl[0]), float(wl[-1]))
            ax2.set_ylim(0.0, 1.0)
            ax2.legend(loc="upper right", frameon=True)
            fig.tight_layout()
            resample_path = os.path.join(output_dir, "resampler_demo_output.png")
            fig.savefig(resample_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            output_paths["resampler_demo_output"] = resample_path

    for key, path in output_paths.items():
        logger.info(f"Additional visualization generated [{key}]: {path}")

    return output_paths
