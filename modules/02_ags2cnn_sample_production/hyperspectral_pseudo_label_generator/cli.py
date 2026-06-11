"""
Command-line interface for hyperspectral pseudo-label generator.
高光谱伪标签生成器的命令行界面。

This module provides a CLI for running the pseudo-label generation pipeline.
此模块提供用于运行伪标签生成流程的CLI。
"""

import argparse
import sys
from pathlib import Path

from .config import ProcessingConfig
from .pipeline import PseudoLabelPipeline
from .logging_config import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.
    创建命令行参数解析器。
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Hyperspectral Pseudo-Label Generator (高光谱伪标签生成器)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (示例):
  # Basic usage with default parameters (使用默认参数的基本用法)
  python -m hyperspectral_pseudo_label_generator.cli \\
    --image data/image.tif \\
    --reference data/reference.npy \\
    --metadata data/wavelengths.json

  # Custom configuration (自定义配置)
  python -m hyperspectral_pseudo_label_generator.cli \\
    --image data/image.tif \\
    --reference data/reference.npy \\
    --metadata data/wavelengths.json \\
    --n-components 10 \\
    --ore-percentile 1.0 \\
    --output-dir ./results

  # Specify output filename (指定输出文件名)
  python -m hyperspectral_pseudo_label_generator.cli \\
    --image data/image.tif \\
    --reference data/reference.npy \\
    --metadata data/wavelengths.json \\
    --output-file my_labels.npy

For more information, see the documentation.
更多信息请参阅文档。
        """
    )
    
    # Required arguments (必需参数)
    required = parser.add_argument_group('required arguments (必需参数)')
    required.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to hyperspectral image file (GeoTIFF or .npy) '
             '(高光谱影像文件路径，GeoTIFF或.npy格式)'
    )
    required.add_argument(
        '--reference', '-r',
        type=str,
        required=True,
        help='Path to reference spectra file (.npy or .csv) '
             '(参考光谱文件路径，.npy或.csv格式)'
    )
    required.add_argument(
        '--metadata', '-m',
        type=str,
        required=True,
        help='Path to wavelength metadata file (JSON or CSV) '
             '(波长元数据文件路径，JSON或CSV格式)'
    )
    
    # Optional arguments (可选参数)
    optional = parser.add_argument_group('optional arguments (可选参数)')
    
    # PCA parameters (PCA参数)
    optional.add_argument(
        '--n-components',
        type=int,
        default=10,
        help='Number of PCA components to select (default: 10) '
             '(选择的PCA成分数量，默认: 10)'
    )
    
    # Thresholding parameters (阈值参数)
    optional.add_argument(
        '--ore-percentile',
        type=float,
        default=17.0,
        help='Percentile threshold for ore classes (default: 17.0) '
             '(矿石类别的百分位数阈值，默认: 17.0)'
    )
    optional.add_argument(
        '--non-ore-percentile',
        type=float,
        default=6.5,
        help='Percentile threshold for non-ore classes (default: 6.5) '
             '(非矿石类别的百分位数阈值，默认: 6.5)'
    )
    optional.add_argument(
        '--poor-percentile',
        type=float,
        default=12.0,
        help='Percentile threshold for poor ore class in 3-class mode (default: 12.0) '
             '(三分类模式下贫矿类别百分位阈值，默认: 12.0)'
    )
    optional.add_argument(
        '--wall-percentile',
        type=float,
        default=8.5,
        help='Percentile threshold for wall rock class in 3-class mode (default: 8.5) '
             '(三分类模式下围岩类别百分位阈值，默认: 8.5)'
    )
    optional.add_argument(
        '--ambiguity-threshold',
        type=float,
        default=2e-7,
        help='Threshold for ambiguity detection (default: 2e-7) '
             '(歧义检测的阈值，默认: 2e-7)'
    )
    
    # Numerical stability (数值稳定性)
    optional.add_argument(
        '--epsilon',
        type=float,
        default=1e-10,
        help='Small value to prevent division by zero (default: 1e-10) '
             '(防止除以零的小值，默认: 1e-10)'
    )
    
    # Memory management (内存管理)
    optional.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Spatial chunk size for processing (default: 1000) '
             '(处理的空间块大小，默认: 1000)'
    )
    
    # Wavelength range (波长范围)
    optional.add_argument(
        '--aloh-min',
        type=float,
        default=2150.0,
        help='Minimum wavelength for Al-OH absorption band in nm (default: 2150.0) '
             '(Al-OH吸收波段的最小波长，纳米，默认: 2150.0)'
    )
    optional.add_argument(
        '--aloh-max',
        type=float,
        default=2250.0,
        help='Maximum wavelength for Al-OH absorption band in nm (default: 2250.0) '
             '(Al-OH吸收波段的最大波长，纳米，默认: 2250.0)'
    )

    optional.add_argument(
        '--amcs-shadow-corr-threshold',
        type=float,
        default=0.14,
        help='AMCS shadow-correlation threshold (default: 0.14) '
             '(AMCS阴影相关性阈值，默认: 0.14)'
    )
    optional.add_argument(
        '--amcs-snr-threshold',
        type=float,
        default=1.8,
        help='AMCS SNR threshold for noise-tail removal (default: 1.8) '
             '(AMCS去尾信噪比阈值，默认: 1.8)'
    )
    optional.add_argument(
        '--amcs-moran-threshold',
        type=float,
        default=0.10,
        help='AMCS Moran I threshold for random-noise removal (default: 0.10) '
             '(AMCS莫兰指数阈值，默认: 0.10)'
    )
    optional.add_argument(
        '--sid-smoothing-window',
        type=int,
        default=3,
        help='Window size for SID spatial mean filtering (default: 3) '
             '(SID空间均值滤波窗口大小，默认: 3)'
    )
    optional.add_argument(
        '--use-illumination-invariant-sid',
        action='store_true',
        default=True,
        help='Enable illumination-invariant SID branch before fusion (default: enabled) '
             '(启用光照不变SID分支并融合，默认: 启用)'
    )
    optional.add_argument(
        '--disable-illumination-invariant-sid',
        action='store_false',
        dest='use_illumination_invariant_sid',
        help='Disable illumination-invariant SID branch (关闭光照不变SID分支)'
    )
    optional.add_argument(
        '--sid-invariant-weight',
        type=float,
        default=0.6,
        help='Fusion weight for illumination-invariant SID (default: 0.6) '
             '(光照不变SID融合权重，默认: 0.6)'
    )
    optional.add_argument(
        '--sid-disagreement-penalty',
        type=float,
        default=0.6,
        help='Penalty factor for disagreement between raw/invariant SID (default: 0.6) '
             '(原始/光照不变SID差异惩罚系数，默认: 0.6)'
    )
    optional.add_argument(
        '--pre-shadow-global-percentile',
        type=float,
        default=25.0,
        help='Pre-classification global shadow percentile (default: 25.0) '
             '(分类前全局阴影百分位阈值，默认: 25.0)'
    )
    optional.add_argument(
        '--pre-shadow-local-percentile',
        type=float,
        default=25.0,
        help='Pre-classification local shadow percentile (default: 25.0) '
             '(分类前局部阴影百分位阈值，默认: 25.0)'
    )
    optional.add_argument(
        '--pre-edge-percentile',
        type=float,
        default=90.0,
        help='Pre-classification edge percentile for unreliable mask (default: 90.0) '
             '(分类前不可靠掩膜的边缘百分位阈值，默认: 90.0)'
    )
    optional.add_argument(
        '--pre-unreliable-dilation-radius',
        type=int,
        default=2,
        help='Dilation radius for pre-unreliable mask (default: 2) '
             '(分类前不可靠掩膜膨胀半径，默认: 2)'
    )
    optional.add_argument(
        '--shadow-exclusion-percentile',
        type=float,
        default=22.0,
        help='Percentile of darkest pixels to suppress from labeled results (default: 22.0) '
             '(在标注结果中剔除最暗像素的百分位，默认: 22.0)'
    )
    optional.add_argument(
        '--shadow-local-percentile',
        type=float,
        default=22.0,
        help='Percentile on local illumination-normalized grayscale for shadow suppression (default: 22.0) '
             '(局部光照归一化灰度阴影抑制百分位，默认: 22.0)'
    )
    optional.add_argument(
        '--shadow-local-window-size',
        type=int,
        default=151,
        help='Local window size for illumination normalization in shadow suppression (default: 151) '
             '(阴影抑制的局部光照归一窗口大小，默认: 151)'
    )
    optional.add_argument(
        '--edge-exclusion-percentile',
        type=float,
        default=88.0,
        help='Percentile of strongest grayscale gradients excluded from labels (default: 88.0) '
             '(从标注中剔除灰度强梯度像素的百分位，默认: 88.0)'
    )
    optional.add_argument(
        '--rich-min-neighbors',
        type=int,
        default=1,
        help='Minimum 8-neighborhood count for rich labels (default: 1) '
             '(富矿标签最小8邻域邻居数，默认: 1)'
    )
    optional.add_argument(
        '--poor-min-neighbors',
        type=int,
        default=1,
        help='Minimum 8-neighborhood count for poor labels (default: 1) '
             '(贫矿标签最小8邻域邻居数，默认: 1)'
    )
    optional.add_argument(
        '--rich-core-window-size',
        type=int,
        default=25,
        help='Window size for rich-core density estimation (default: 25) '
             '(富矿核心密度估计窗口大小，默认: 25)'
    )
    optional.add_argument(
        '--rich-core-min-density',
        type=float,
        default=0.015,
        help='Minimum local rich density for rich-core retention (default: 0.015) '
             '(富矿核心保留的最小局部富矿密度，默认: 0.015)'
    )
    optional.add_argument(
        '--poor-near-rich-radius',
        type=int,
        default=12,
        help='Spatial radius requiring poor labels to be near rich centers (default: 12) '
             '(贫矿标签需靠近富矿中心的空间半径，默认: 12)'
    )
    optional.add_argument(
        '--rich-to-poor-sid-margin',
        type=float,
        default=-3e-4,
        help='SID margin for converting non-core rich labels to poor labels (default: -3e-4) '
             '(将非核心富矿转为贫矿的SID差值阈值，默认: -3e-4)'
    )
    optional.add_argument(
        '--poor-confidence-percentile',
        type=float,
        default=2.0,
        help='Percentile of highly confident poor labels kept even if far from rich centers (default: 2.0) '
             '(即使远离富矿中心也保留的高置信贫矿百分位，默认: 2.0)'
    )
    
    # Output options (输出选项)
    optional.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output) '
             '(结果输出目录，默认: ./output)'
    )
    optional.add_argument(
        '--output-file',
        type=str,
        default='pseudo_label_map.npy',
        help='Output filename for pseudo-label map (default: pseudo_label_map.npy) '
             '(伪标签图的输出文件名，默认: pseudo_label_map.npy)'
    )
    
    # Logging options (日志选项)
    optional.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO) '
             '(日志级别，默认: INFO)'
    )
    
    return parser


def validate_paths(args: argparse.Namespace) -> None:
    """
    Validate that input file paths exist.
    验证输入文件路径存在。
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        FileNotFoundError: If any input file does not exist
    """
    # Check image file
    # 检查影像文件
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")
    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {args.image}")
    
    # Check reference file
    # 检查参考文件
    reference_path = Path(args.reference)
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference}")
    if not reference_path.is_file():
        raise ValueError(f"Reference path is not a file: {args.reference}")
    
    # Check metadata file
    # 检查元数据文件
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")
    if not metadata_path.is_file():
        raise ValueError(f"Metadata path is not a file: {args.metadata}")


def create_config_from_args(args: argparse.Namespace) -> ProcessingConfig:
    """
    Create ProcessingConfig from command-line arguments.
    从命令行参数创建ProcessingConfig。
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        ProcessingConfig instance
    """
    config = ProcessingConfig(
        n_components=args.n_components,
        ore_percentile=args.ore_percentile,
        non_ore_percentile=args.non_ore_percentile,
        poor_percentile=args.poor_percentile,
        wall_percentile=args.wall_percentile,
        ambiguity_threshold=args.ambiguity_threshold,
        epsilon=args.epsilon,
        chunk_size=args.chunk_size,
        aloh_min_wavelength=args.aloh_min,
        aloh_max_wavelength=args.aloh_max,
        amcs_shadow_corr_threshold=args.amcs_shadow_corr_threshold,
        amcs_snr_threshold=args.amcs_snr_threshold,
        amcs_moran_threshold=args.amcs_moran_threshold,
        sid_smoothing_window=args.sid_smoothing_window,
        use_illumination_invariant_sid=args.use_illumination_invariant_sid,
        sid_invariant_weight=args.sid_invariant_weight,
        sid_disagreement_penalty=args.sid_disagreement_penalty,
        pre_shadow_global_percentile=args.pre_shadow_global_percentile,
        pre_shadow_local_percentile=args.pre_shadow_local_percentile,
        pre_edge_percentile=args.pre_edge_percentile,
        pre_unreliable_dilation_radius=args.pre_unreliable_dilation_radius,
        shadow_exclusion_percentile=args.shadow_exclusion_percentile,
        shadow_local_percentile=args.shadow_local_percentile,
        shadow_local_window_size=args.shadow_local_window_size,
        edge_exclusion_percentile=args.edge_exclusion_percentile,
        rich_min_neighbors=args.rich_min_neighbors,
        poor_min_neighbors=args.poor_min_neighbors,
        rich_core_window_size=args.rich_core_window_size,
        rich_core_min_density=args.rich_core_min_density,
        poor_near_rich_radius=args.poor_near_rich_radius,
        rich_to_poor_sid_margin=args.rich_to_poor_sid_margin,
        poor_confidence_percentile=args.poor_confidence_percentile,
        output_dir=args.output_dir
    )
    
    return config


def main(argv=None):
    """
    Main entry point for command-line interface.
    命令行界面的主入口点。
    
    Args:
        argv: Command-line arguments (default: sys.argv)
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse arguments
    # 解析参数
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    # 设置日志
    setup_logging(level=args.log_level)
    
    try:
        # Validate input paths
        # 验证输入路径
        validate_paths(args)
        
        # Create configuration
        # 创建配置
        config = create_config_from_args(args)
        
        # Create and run pipeline
        # 创建并运行流程
        pipeline = PseudoLabelPipeline(config)
        pipeline.run(
            image_path=args.image,
            reference_path=args.reference,
            metadata_path=args.metadata,
            output_filename=args.output_file
        )
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("Please check that all input files exist.", file=sys.stderr)
        return 1
        
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("Please check your input parameters and data.", file=sys.stderr)
        return 2
        
    except IOError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("Please check file permissions and disk space.", file=sys.stderr)
        return 3
        
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        print("Please report this issue with the full error message.", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

