#!/usr/bin/env python3
"""
Main pipeline script for building Ground Standard Reference Spectral Library (GSRSL)
构建地面标准参考光谱库(GSRSL)的主管道脚本

================================================================================
重要说明：如何设置输入和输出路径
================================================================================

本脚本使用命令行参数来指定所有路径，不需要在脚本内部修改代码。

【必需的输入参数】：
  --metadata <路径>    GF-5卫星元数据文件路径
  --spectra <路径>     地面光谱文件所在目录路径
  --labels <路径>      标签表CSV文件路径（包含filename和class_id两列）
  --output <路径>      输出目录路径

【可选参数】：
  --log <路径>         日志文件路径（默认：gsrsl_pipeline.log）
  --verbose            启用详细日志输出
  --pattern <模式>     光谱文件匹配模式（默认：*.csv）

【输出文件】（自动生成在--output指定的目录中）：
  ├─ gsrsl.npy                    GSRSL数据文件（NumPy格式）
  ├─ gsrsl_visualization.png      可视化图像
  └─ gsrsl_pipeline.log           日志文件（如果未指定--log参数）

【使用示例】：

  # 基本用法
  python build_gsrsl.py \
      --metadata data/gf5_metadata.txt \
      --spectra data/ground_spectra/ \
      --labels data/labels.csv \
      --output output/

  # 使用自定义日志路径
  python build_gsrsl.py \
      --metadata D:/MyData/gf5_metadata.txt \
      --spectra D:/MyData/spectra/ \
      --labels D:/MyData/labels.csv \
      --output D:/Results/ \
      --log D:/Results/my_pipeline.log

  # 启用详细日志
  python build_gsrsl.py \
      --metadata data/gf5_metadata.txt \
      --spectra data/ground_spectra/ \
      --labels data/labels.csv \
      --output output/ \
      --verbose

================================================================================

This script orchestrates the complete GSRSL data processing pipeline:
该脚本编排完整的GSRSL数据处理管道：

1. Parse GF-5 satellite metadata to extract band specifications
   解析GF-5卫星元数据以提取波段规格
2. Load ground spectral measurements from TSG8 files
   从TSG8文件加载地面光谱测量
3. Apply Savitzky-Golay denoising filter
   应用Savitzky-Golay去噪滤波器
4. Resample ground spectra to satellite band resolution using Gaussian SRF
   使用高斯SRF将地面光谱重采样到卫星波段分辨率
5. Aggregate spectra by lithology class
   按岩性类别聚合光谱
6. Save GSRSL to file and generate visualization
   将GSRSL保存到文件并生成可视化

Usage:
    python build_gsrsl.py --metadata <path> --spectra <dir> --labels <path> --output <dir>

Example:
    python build_gsrsl.py \\
        --metadata data/gf5_metadata.txt \\
        --spectra data/ground_spectra/ \\
        --labels data/labels.csv \\
        --output output/

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.3
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

# Import GSRSL pipeline modules
from gsrsl_pipeline.logging_config import setup_logging, get_logger
from gsrsl_pipeline.metadata_parser import parse_metadata
from gsrsl_pipeline.spectrum_loader import load_ground_spectrum, load_spectra_from_matrix
from gsrsl_pipeline.filters import apply_savgol_filter
from gsrsl_pipeline.resampler import SpectralResampler
from gsrsl_pipeline.aggregator import aggregate_by_class
from gsrsl_pipeline.output import save_gsrsl, visualize_gsrsl, generate_additional_visualizations


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    解析命令行参数
    
    ============================================================================
    重要说明：如何设置输入和输出路径
    ============================================================================
    
    本脚本通过命令行参数接收所有路径配置，不需要在脚本内部修改代码。
    
    【三个必需的输入路径】：
    
    1. --metadata <路径>
       GF-5卫星元数据文件路径
       示例：--metadata data/gf5_metadata.txt
       文件格式：包含波长和FWHM值的文本文件
    
    2. --spectra <路径>
       地面光谱矩阵CSV文件路径
       示例：--spectra data/ground_spectra_matrix.csv
       文件格式：第一列为波长(nm)，后续每一列为一个样本的反射率序列
    
    3. --labels <路径>
       标签表文件路径（CSV格式）
       示例：--labels data/labels.csv
       文件格式：必须包含两列 'filename' 和 'class_id'
    
    【一个必需的输出路径】：
    
    4. --output <路径>
       输出目录路径，用于保存所有输出文件
       示例：--output output/
       将生成以下文件：
       - gsrsl.npy (GSRSL数据文件)
       - gsrsl_visualization.png (可视化图像)
    
    【可选参数】：
    
    5. --log <路径>
       日志文件保存路径（可选，默认：gsrsl_pipeline.log）
       示例：--log logs/my_pipeline.log
    
    6. --verbose
       启用详细日志输出（可选）
    
    7. --pattern <模式>
       光谱文件匹配模式（可选，默认：*.csv）
       示例：--pattern *.txt
    
    ============================================================================
    使用示例（在命令行中运行）：
    ============================================================================
    
    # Windows系统示例：
    python build_gsrsl.py --metadata data/gf5_metadata.txt --spectra data/ground_spectra/ --labels data/labels.csv --output output/
    
    # 或者使用自定义日志路径：
    python build_gsrsl.py --metadata D:/MyData/gf5_metadata.txt --spectra D:/MyData/spectra/ --labels D:/MyData/labels.csv --output D:/Results/ --log D:/Results/my_log.log
    
    ============================================================================
    
    Returns:
        Parsed arguments namespace
        解析的参数命名空间
    """
    parser = argparse.ArgumentParser(
        description='Build Ground Standard Reference Spectral Library (GSRSL) for GF-5 satellite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python build_gsrsl.py \\
      --metadata data/gf5_metadata.txt \\
      --spectra data/ground_spectra/ \\
      --labels data/labels.csv \\
      --output output/

  # With custom log file
  python build_gsrsl.py \\
      --metadata data/gf5_metadata.txt \\
      --spectra data/ground_spectra/ \\
      --labels data/labels.csv \\
      --output output/ \\
      --log pipeline.log

  # With verbose logging
  python build_gsrsl.py \\
      --metadata data/gf5_metadata.txt \\
      --spectra data/ground_spectra/ \\
      --labels data/labels.csv \\
      --output output/ \\
      --verbose

Lithology Classes:
  0: Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)
  1: Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)
  2: Mixed-type Rich Pegmatite (混合型富矿伟晶岩)
  3: Barren Pegmatite (贫矿伟晶岩)
  4: Wall Rock (围岩)
        """
    )
    
    # ========================================================================
    # 必需参数 - Required arguments
    # ========================================================================
    
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='【输入1】GF-5元数据文件路径 | Path to GF-5 metadata text file containing wavelengths and FWHM values'
    )
    
    parser.add_argument(
        '--spectra',
        type=str,
        required=True,
        help='【输入2】地面光谱矩阵CSV文件路径 | Path to CSV file containing ground spectra in matrix format (column 0 = wavelength, columns 1..N = reflectance per sample from TSG8)'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='【输入3】标签表文件路径 | Path to CSV file mapping filenames to class IDs (columns: filename, class_id)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='【输出目录】输出文件保存目录 | Path to output directory for GSRSL file and visualization'
    )
    
    # ========================================================================
    # 可选参数 - Optional arguments
    # ========================================================================
    
    parser.add_argument(
        '--log',
        type=str,
        default='gsrsl_pipeline.log',
        help='【可选】日志文件路径（默认：gsrsl_pipeline.log）| Path to log file (default: gsrsl_pipeline.log)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='【可选】启用详细日志 | Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.csv',
        help='【可选】光谱文件匹配模式（默认：*.csv）| File pattern for spectrum files (default: *.csv)'
    )
    
    return parser.parse_args()


def validate_input_paths(args: argparse.Namespace, logger) -> None:
    """
    Validate that all input paths exist.
    验证所有输入路径是否存在
    
    Args:
        args: Parsed command-line arguments
             解析的命令行参数
        logger: Logger instance
               日志记录器实例
    
    Raises:
        FileNotFoundError: If any required input file or directory doesn't exist
                          如果任何必需的输入文件或目录不存在
    """
    logger.info("Validating input paths...")
    
    # Validate metadata file
    if not os.path.exists(args.metadata):
        raise FileNotFoundError(
            f"Metadata file not found: '{args.metadata}'. "
            f"Please verify the path is correct."
        )
    logger.info(f"✓ Metadata file found: {args.metadata}")
    
    # Validate spectra matrix file
    if not os.path.exists(args.spectra):
        raise FileNotFoundError(
            f"Spectra file not found: '{args.spectra}'. "
            f"Please verify the path is correct."
        )
    if not os.path.isfile(args.spectra):
        raise FileNotFoundError(
            f"Spectra path is not a file: '{args.spectra}'. "
            f"Please provide a CSV file containing ground spectra in matrix format."
        )
    logger.info(f"✓ Spectra file found: {args.spectra}")
    
    # Validate labels file
    if not os.path.exists(args.labels):
        raise FileNotFoundError(
            f"Labels file not found: '{args.labels}'. "
            f"Please verify the path is correct."
        )
    logger.info(f"✓ Labels file found: {args.labels}")
    
    logger.info("All input paths validated successfully")


def load_label_table(labels_path: str, logger) -> pd.DataFrame:
    """
    Load the label table mapping filenames to class IDs.
    加载将文件名映射到类别ID的标签表
    
    Args:
        labels_path: Path to labels CSV file
                    标签CSV文件的路径
        logger: Logger instance
               日志记录器实例
    
    Returns:
        DataFrame with columns ['filename', 'class_id']
        包含列['filename', 'class_id']的DataFrame
    
    Raises:
        ValueError: If labels file format is invalid
                   如果标签文件格式无效
    """
    logger.info(f"Loading label table from {labels_path}...")
    
    try:
        label_table = pd.read_csv(labels_path)
    except Exception as e:
        raise ValueError(
            f"Failed to read labels file '{labels_path}': {e}. "
            f"Expected CSV format with columns 'filename' and 'class_id'."
        ) from e
    
    # Validate required columns
    if 'filename' not in label_table.columns:
        raise ValueError(
            f"Labels file must contain 'filename' column. "
            f"Found columns: {list(label_table.columns)}"
        )
    
    if 'class_id' not in label_table.columns:
        raise ValueError(
            f"Labels file must contain 'class_id' column. "
            f"Found columns: {list(label_table.columns)}"
        )
    
    # Validate class IDs are in valid range
    invalid_classes = label_table[
        (label_table['class_id'] < 0) | (label_table['class_id'] > 4)
    ]
    if len(invalid_classes) > 0:
        raise ValueError(
            f"Found {len(invalid_classes)} invalid class IDs. "
            f"Class IDs must be in range [0, 4]. "
            f"Invalid entries: {invalid_classes[['filename', 'class_id']].to_dict('records')}"
        )
    
    # Log statistics
    class_counts = label_table['class_id'].value_counts().sort_index()
    logger.info(f"Label table loaded: {len(label_table)} samples")
    for class_id, count in class_counts.items():
        logger.info(f"  Class {class_id}: {count} samples")
    
    return label_table


def find_spectrum_files(spectra_dir: str, pattern: str, logger) -> List[str]:
    """
    Find all spectrum files in the directory matching the pattern.
    查找目录中匹配模式的所有光谱文件
    
    Args:
        spectra_dir: Path to directory containing spectrum files
                    包含光谱文件的目录路径
        pattern: File pattern (e.g., '*.csv', '*.txt')
                文件模式（例如，'*.csv'，'*.txt'）
        logger: Logger instance
               日志记录器实例
    
    Returns:
        List of full paths to spectrum files
        光谱文件的完整路径列表
    
    Raises:
        ValueError: If no spectrum files are found
                   如果未找到光谱文件
    """
    logger.info(f"Searching for spectrum files in {spectra_dir} with pattern '{pattern}'...")
    
    spectra_path = Path(spectra_dir)
    spectrum_files = sorted(spectra_path.glob(pattern))
    
    if len(spectrum_files) == 0:
        raise ValueError(
            f"No spectrum files found in '{spectra_dir}' matching pattern '{pattern}'. "
            f"Please verify the directory contains spectrum files."
        )
    
    logger.info(f"Found {len(spectrum_files)} spectrum files")
    
    return [str(f) for f in spectrum_files]


def main():
    """
    Main pipeline execution function.
    主管道执行函数
    
    ============================================================================
    路径配置总结
    ============================================================================
    
    本脚本通过命令行参数接收所有路径，无需修改代码。
    
    【输入路径】（3个必需参数）：
    1. args.metadata  - GF-5元数据文件路径
    2. args.spectra   - 地面光谱矩阵CSV文件路径
    3. args.labels    - 标签表CSV文件路径
    
    【输出路径】（1个必需参数 + 1个可选参数）：
    4. args.output    - 输出目录路径（必需）
       └─ gsrsl.npy                    （自动生成）
       └─ gsrsl_visualization.png      （自动生成）
    5. args.log       - 日志文件路径（可选，默认：gsrsl_pipeline.log）
    
    【命令行使用示例】：
    
    python build_gsrsl.py \
        --metadata data/gf5_metadata.txt \
        --spectra data/ground_spectra/ \
        --labels data/labels.csv \
        --output output/ \
        --log logs/pipeline.log
    
    ============================================================================
    """
    # Parse command-line arguments
    # 解析命令行参数（所有路径都从这里获取）
    args = parse_arguments()
    
    # Setup logging
    log_level = 10 if args.verbose else 20  # DEBUG if verbose, else INFO
    logger = setup_logging(args.log, level=log_level)
    
    logger.info("=" * 80)
    logger.info("GSRSL Pipeline Started")
    logger.info("地面标准参考光谱库管道已启动")
    logger.info("=" * 80)
    
    try:
        # Step 1: Validate input paths
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Validating Input Paths")
        logger.info("步骤1：验证输入路径")
        logger.info("=" * 80)
        validate_input_paths(args, logger)
        
        # Step 2: Parse GF-5 metadata
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Parsing GF-5 Satellite Metadata")
        logger.info("步骤2：解析GF-5卫星元数据")
        logger.info("=" * 80)
        sat_wavelengths, sat_fwhms = parse_metadata(args.metadata)
        logger.info(
            f"Successfully parsed metadata: 297 bands, "
            f"wavelength range [{sat_wavelengths[0]:.2f}, {sat_wavelengths[-1]:.2f}] nm"
        )
        
        # Step 3: Load label table
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Loading Label Table")
        logger.info("步骤3：加载标签表")
        logger.info("=" * 80)
        label_table = load_label_table(args.labels, logger)
        
        # Step 4: Load ground spectra from matrix file
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Loading Ground Spectra From Matrix File")
        logger.info("步骤4：从矩阵CSV文件加载地面光谱")
        logger.info("=" * 80)
        ground_spectra = load_spectra_from_matrix(args.spectra, label_table)
        
        # Step 5: Initialize spectral resampler
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Initializing Spectral Resampler")
        logger.info("步骤5：初始化光谱重采样器")
        logger.info("=" * 80)
        resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
        
        # Step 6: Process each spectrum
        logger.info("\n" + "=" * 80)
        logger.info("Step 6: Processing Ground Spectra")
        logger.info("步骤6：处理地面光谱")
        logger.info("=" * 80)
        
        resampled_spectra = []
        example_payload = {}
        n_processed = 0
        n_anomalous = 0
        n_failed = 0
        
        for i, ground_spectrum in enumerate(ground_spectra, 1):
            try:
                logger.info(
                    f"\nProcessing spectrum {i}/{len(ground_spectra)}: {ground_spectrum.filename}"
                )
                
                # Track anomalous spectra
                if ground_spectrum.is_anomalous:
                    n_anomalous += 1
                
                # Apply Savitzky-Golay filter
                filtered_spectrum = apply_savgol_filter(ground_spectrum)
                logger.debug(f"  Applied Savitzky-Golay filter")
                
                # Resample to satellite bands
                resampled = resampler.resample(filtered_spectrum)
                logger.debug(f"  Resampled to 297 satellite bands")
                
                # Store result
                resampled_spectra.append((ground_spectrum.class_id, resampled))
                if ground_spectrum.class_id not in example_payload:
                    example_payload[ground_spectrum.class_id] = {
                        "filename": ground_spectrum.filename,
                        "wavelengths": ground_spectrum.wavelengths.copy(),
                        "reflectances": ground_spectrum.reflectances.copy(),
                        "filtered_reflectances": filtered_spectrum.reflectances.copy(),
                        "resampled_reflectances": resampled.copy()
                    }
                n_processed += 1
                
                # Log progress every 10 spectra
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(ground_spectra)} spectra processed")
                
            except Exception as e:
                n_failed += 1
                logger.error(f"Failed to process spectrum '{ground_spectrum.filename}': {e}")
                if args.verbose:
                    logger.exception("Detailed error:")
                continue
        
        # Log processing summary
        logger.info("\n" + "-" * 80)
        logger.info("Processing Summary:")
        logger.info(f"  Total spectra: {len(ground_spectra)}")
        logger.info(f"  Successfully processed: {n_processed}")
        logger.info(f"  Anomalous spectra: {n_anomalous}")
        logger.info(f"  Failed: {n_failed}")
        logger.info("-" * 80)
        
        if n_processed == 0:
            raise ValueError(
                "No spectra were successfully processed. "
                "Please check the input files and error messages above."
            )
        
        # Step 7: Aggregate by class
        logger.info("\n" + "=" * 80)
        logger.info("Step 7: Aggregating Spectra by Lithology Class")
        logger.info("步骤7：按岩性类别聚合光谱")
        logger.info("=" * 80)
        class_means = aggregate_by_class(resampled_spectra)
        
        # Step 8: Save GSRSL
        logger.info("\n" + "=" * 80)
        logger.info("Step 8: Saving GSRSL to File")
        logger.info("步骤8：将GSRSL保存到文件")
        logger.info("=" * 80)
        
        # ====================================================================
        # 输出文件路径说明
        # ====================================================================
        # 所有输出文件都保存在 args.output 指定的目录中
        # 该目录通过命令行参数 --output 设置
        # 
        # 输出文件1：GSRSL数据文件
        # 文件名：gsrsl.npy（固定名称）
        # 完整路径：<output目录>/gsrsl.npy
        # 内容：5个岩性类别的平均光谱数据（NumPy数组格式）
        # ====================================================================
        
        # Create output directory if needed
        # 如果输出目录不存在，自动创建
        os.makedirs(args.output, exist_ok=True)
        
        # Save GSRSL
        # 保存GSRSL数据到 <output目录>/gsrsl.npy
        gsrsl_path = os.path.join(args.output, 'gsrsl.npy')
        save_gsrsl(class_means, gsrsl_path)
        
        # Step 9: Generate visualization
        logger.info("\n" + "=" * 80)
        logger.info("Step 9: Generating Visualization")
        logger.info("步骤9：生成可视化")
        logger.info("=" * 80)
        
        # ====================================================================
        # 输出文件2：可视化图像
        # 文件名：gsrsl_visualization.png（固定名称）
        # 完整路径：<output目录>/gsrsl_visualization.png
        # 内容：5个岩性类别光谱曲线的可视化图表
        # ====================================================================
        
        viz_path = os.path.join(args.output, 'gsrsl_visualization.png')
        visualize_gsrsl(class_means, sat_wavelengths, viz_path)
        
        extra_visualizations = generate_additional_visualizations(
            ground_spectra=ground_spectra,
            sat_wavelengths=sat_wavelengths,
            class_means=class_means,
            example_payload=example_payload,
            output_dir=args.output
        )
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("GSRSL Pipeline Completed Successfully!")
        logger.info("GSRSL管道成功完成！")
        logger.info("=" * 80)
        logger.info("\nOutput Files:")
        logger.info(f"  GSRSL data: {gsrsl_path}")
        logger.info(f"  Visualization: {viz_path}")
        logger.info("  Additional visualizations:")
        for name, path in extra_visualizations.items():
            logger.info(f"    {name}: {path}")
        logger.info(f"  Log file: {args.log}")
        logger.info("\nYou can now use the GSRSL for satellite data processing and deep learning.")
        logger.info("您现在可以使用GSRSL进行卫星数据处理和深度学习。")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted by user (Ctrl+C)")
        return 130
        
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("GSRSL Pipeline Failed!")
        logger.error("GSRSL管道失败！")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        if args.verbose:
            logger.exception("Detailed error traceback:")
        logger.error("\nPlease check the error messages above and verify your input files.")
        logger.error("请检查上面的错误消息并验证您的输入文件。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
