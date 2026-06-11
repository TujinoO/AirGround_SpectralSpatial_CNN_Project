"""
Complete pipeline example for hyperspectral pseudo-label generator.
高光谱伪标签生成器的完整流程示例。

This example demonstrates how to use the full pipeline with all components.
此示例演示如何使用包含所有组件的完整流程。
"""

import numpy as np
import os
from pathlib import Path

from hyperspectral_pseudo_label_generator import ProcessingConfig, PseudoLabelPipeline
from hyperspectral_pseudo_label_generator.logging_config import setup_logging, get_logger


def create_synthetic_data(output_dir: str = "./example_data"):
    """
    Create synthetic test data for demonstration.
    创建用于演示的合成测试数据。
    
    Args:
        output_dir: Directory to save synthetic data
    """
    logger = get_logger(__name__)
    logger.info("Creating synthetic test data...")
    logger.info("创建合成测试数据...")
    
    # Create output directory
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic hyperspectral image (small for demo)
    # 创建合成高光谱影像（演示用小尺寸）
    H, W = 50, 50  # Small image for quick processing
    bands = 297
    
    logger.info(f"  Creating synthetic image: {H}x{W}x{bands}")
    
    # Generate random reflectance values
    # 生成随机反射率值
    np.random.seed(42)
    image = np.random.rand(H, W, bands).astype(np.float32) * 0.5 + 0.2
    
    # Add some spatial structure
    # 添加一些空间结构
    for i in range(H):
        for j in range(W):
            # Create regions with different spectral characteristics
            # 创建具有不同光谱特征的区域
            if i < H // 3:
                image[i, j, :] *= 1.2  # Brighter region
            elif i > 2 * H // 3:
                image[i, j, :] *= 0.8  # Darker region
    
    image_path = os.path.join(output_dir, "synthetic_image.npy")
    np.save(image_path, image)
    logger.info(f"  Saved image to: {image_path}")
    
    # Create synthetic reference spectra (5 classes)
    # 创建合成参考光谱（5个类别）
    logger.info("  Creating synthetic reference spectra...")
    reference_spectra = np.zeros((5, bands), dtype=np.float32)
    
    for class_id in range(5):
        # Each class has a different spectral signature
        # 每个类别有不同的光谱特征
        base_spectrum = np.random.rand(bands).astype(np.float32) * 0.3 + 0.3
        
        # Add characteristic absorption features
        # 添加特征吸收特征
        if class_id in [0, 1, 2]:  # Ore classes
            # Strong absorption around band 200 (simulating Al-OH)
            # 在波段200附近有强吸收（模拟Al-OH）
            base_spectrum[190:210] *= 0.5
        elif class_id == 3:  # Barren pegmatite
            # Moderate absorption
            # 中等吸收
            base_spectrum[190:210] *= 0.7
        else:  # Wall rock
            # Weak absorption
            # 弱吸收
            base_spectrum[190:210] *= 0.9
        
        reference_spectra[class_id] = base_spectrum
    
    reference_path = os.path.join(output_dir, "synthetic_reference.npy")
    np.save(reference_path, reference_spectra)
    logger.info(f"  Saved reference spectra to: {reference_path}")
    
    # Create synthetic wavelength metadata
    # 创建合成波长元数据
    logger.info("  Creating synthetic wavelength metadata...")
    
    # Linear wavelength spacing from 400 to 2500 nm
    # 从400到2500 nm的线性波长间隔
    wavelengths = np.linspace(400, 2500, bands, dtype=np.float32)
    
    # Save as CSV
    # 保存为CSV
    metadata_path = os.path.join(output_dir, "synthetic_wavelengths.csv")
    np.savetxt(metadata_path, wavelengths, delimiter=',', fmt='%.2f')
    logger.info(f"  Saved wavelengths to: {metadata_path}")
    
    logger.info("Synthetic data creation complete!")
    logger.info("合成数据创建完成！")
    
    return image_path, reference_path, metadata_path


def main():
    """Main example function."""
    # Set up logging
    # 设置日志
    setup_logging(level='INFO')
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("Hyperspectral Pseudo-Label Generator - Pipeline Example")
    logger.info("高光谱伪标签生成器 - 流程示例")
    logger.info("=" * 60)
    
    # Step 1: Create synthetic data
    # 步骤1: 创建合成数据
    logger.info("\nStep 1: Creating synthetic test data...")
    logger.info("步骤1: 创建合成测试数据...")
    
    image_path, reference_path, metadata_path = create_synthetic_data()
    
    # Step 2: Configure the pipeline
    # 步骤2: 配置流程
    logger.info("\nStep 2: Configuring pipeline...")
    logger.info("步骤2: 配置流程...")
    
    config = ProcessingConfig(
        n_components=5,
        ore_percentile=2.5,
        non_ore_percentile=5.0,
        ambiguity_threshold=0.1,
        chunk_size=1000,
        output_dir="./example_output"
    )
    
    logger.info("Configuration:")
    logger.info(f"  - PCA components: {config.n_components}")
    logger.info(f"  - Ore percentile: {config.ore_percentile}%")
    logger.info(f"  - Non-ore percentile: {config.non_ore_percentile}%")
    logger.info(f"  - Ambiguity threshold: {config.ambiguity_threshold}")
    logger.info(f"  - Output directory: {config.output_dir}")
    
    # Step 3: Create and run pipeline
    # 步骤3: 创建并运行流程
    logger.info("\nStep 3: Running pipeline...")
    logger.info("步骤3: 运行流程...")
    
    try:
        pipeline = PseudoLabelPipeline(config)
        
        pseudo_labels = pipeline.run(
            image_path=image_path,
            reference_path=reference_path,
            metadata_path=metadata_path,
            output_filename="example_pseudo_labels.npy"
        )
        
        # Step 4: Display results
        # 步骤4: 显示结果
        logger.info("\nStep 4: Results summary...")
        logger.info("步骤4: 结果摘要...")
        
        unique, counts = np.unique(pseudo_labels, return_counts=True)
        total_pixels = pseudo_labels.size
        
        label_names = {
            0: "Unlabeled/Ambiguous (未标注/歧义)",
            1: "Ore (矿石)",
            2: "Wall Rock (围岩)",
            3: "Barren Pegmatite (贫矿伟晶岩)"
        }
        
        logger.info("\nFinal label distribution:")
        logger.info("最终标签分布:")
        for label, count in zip(unique, counts):
            percentage = count / total_pixels * 100
            logger.info(f"  {label_names.get(label, f'Label {label}')}: "
                       f"{count} pixels ({percentage:.2f}%)")
        
        # Check output files
        # 检查输出文件
        logger.info("\nOutput files:")
        logger.info("输出文件:")
        
        output_files = [
            "example_pseudo_labels.npy",
            "pseudo_label_visualization.png"
        ]
        
        for filename in output_files:
            filepath = os.path.join(config.output_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                logger.info(f"  ✓ {filename} ({size:,} bytes)")
            else:
                logger.warning(f"  ✗ {filename} (not found)")
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline example completed successfully!")
        logger.info("流程示例成功完成！")
        logger.info("=" * 60)
        
        logger.info("\nNext steps:")
        logger.info("下一步:")
        logger.info("  1. Check the output directory for results")
        logger.info("     检查输出目录中的结果")
        logger.info("  2. View the visualization PNG file")
        logger.info("     查看可视化PNG文件")
        logger.info("  3. Load the pseudo-label map for training")
        logger.info("     加载伪标签图用于训练")
        
    except Exception as e:
        logger.error(f"\nPipeline failed: {e}")
        logger.error(f"流程失败: {e}")
        raise


if __name__ == "__main__":
    main()

