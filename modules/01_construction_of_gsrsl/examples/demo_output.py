"""
Demo script for output generator module.
输出生成器模块的演示脚本

This script demonstrates how to use the save_gsrsl() and visualize_gsrsl()
functions to save the Ground Standard Reference Spectral Library (GSRSL)
and generate visualizations.

该脚本演示如何使用save_gsrsl()和visualize_gsrsl()函数保存地面标准参考光谱库(GSRSL)
并生成可视化。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import gsrsl_pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gsrsl_pipeline.output import save_gsrsl, visualize_gsrsl
from gsrsl_pipeline.logging_config import setup_logging


def create_synthetic_gsrsl():
    """
    Create synthetic GSRSL data for demonstration.
    创建用于演示的合成GSRSL数据
    
    Returns:
        Tuple of (class_means, wavelengths)
        返回元组(class_means, wavelengths)
    """
    print("Creating synthetic GSRSL data...")
    print("创建合成GSRSL数据...")
    
    # Create wavelengths array (297 bands from 400-2500 nm)
    # 创建波长数组(297个波段，从400-2500 nm)
    wavelengths = np.linspace(400, 2500, 297, dtype=np.float64)
    
    # Create synthetic spectra for each lithology class
    # 为每个岩性类别创建合成光谱
    class_means = {}
    
    # Class 0: Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)
    # Higher reflectance in visible, absorption around 1400nm and 1900nm
    # 可见光区域反射率较高，在1400nm和1900nm附近有吸收
    spectrum0 = np.full(297, 0.45, dtype=np.float32)
    spectrum0[wavelengths < 800] = 0.55  # Higher in visible
    spectrum0[np.abs(wavelengths - 1400) < 100] *= 0.7  # Water absorption
    spectrum0[np.abs(wavelengths - 1900) < 100] *= 0.6  # Water absorption
    spectrum0[np.abs(wavelengths - 2200) < 50] *= 0.8   # Al-OH absorption
    class_means[0] = spectrum0
    
    # Class 1: Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)
    # Similar to class 0 but with stronger Al-OH absorption
    # 与类别0相似，但Al-OH吸收更强
    spectrum1 = np.full(297, 0.42, dtype=np.float32)
    spectrum1[wavelengths < 800] = 0.52
    spectrum1[np.abs(wavelengths - 1400) < 100] *= 0.7
    spectrum1[np.abs(wavelengths - 1900) < 100] *= 0.6
    spectrum1[np.abs(wavelengths - 2200) < 50] *= 0.65  # Stronger Al-OH
    class_means[1] = spectrum1
    
    # Class 2: Mixed-type Rich Pegmatite (混合型富矿伟晶岩)
    # Intermediate characteristics
    # 中间特征
    spectrum2 = np.full(297, 0.40, dtype=np.float32)
    spectrum2[wavelengths < 800] = 0.50
    spectrum2[np.abs(wavelengths - 1400) < 100] *= 0.75
    spectrum2[np.abs(wavelengths - 1900) < 100] *= 0.65
    spectrum2[np.abs(wavelengths - 2200) < 50] *= 0.75
    class_means[2] = spectrum2
    
    # Class 3: Barren Pegmatite (贫矿伟晶岩)
    # Lower overall reflectance, weaker absorption features
    # 整体反射率较低，吸收特征较弱
    spectrum3 = np.full(297, 0.35, dtype=np.float32)
    spectrum3[wavelengths < 800] = 0.42
    spectrum3[np.abs(wavelengths - 1400) < 100] *= 0.85
    spectrum3[np.abs(wavelengths - 1900) < 100] *= 0.80
    spectrum3[np.abs(wavelengths - 2200) < 50] *= 0.85
    class_means[3] = spectrum3
    
    # Class 4: Wall Rock (围岩)
    # Lowest reflectance, minimal absorption features
    # 反射率最低，吸收特征最少
    spectrum4 = np.full(297, 0.25, dtype=np.float32)
    spectrum4[wavelengths < 800] = 0.30
    spectrum4[np.abs(wavelengths - 1400) < 100] *= 0.90
    spectrum4[np.abs(wavelengths - 1900) < 100] *= 0.85
    class_means[4] = spectrum4
    
    print(f"Created synthetic GSRSL with {len(class_means)} classes")
    print(f"创建了包含{len(class_means)}个类别的合成GSRSL")
    print(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    print(f"波长范围: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    print()
    
    return class_means, wavelengths


def demo_save_gsrsl(class_means, output_dir='examples'):
    """
    Demonstrate saving GSRSL to file.
    演示将GSRSL保存到文件
    
    Args:
        class_means: Dictionary of class mean spectra
                    类别平均光谱的字典
        output_dir: Output directory
                   输出目录
    """
    print("=" * 70)
    print("Demo 1: Saving GSRSL to file")
    print("演示1: 将GSRSL保存到文件")
    print("=" * 70)
    
    # Define output path
    # 定义输出路径
    output_path = os.path.join(output_dir, 'demo_gsrsl.npy')
    
    print(f"Saving GSRSL to: {output_path}")
    print(f"保存GSRSL到: {output_path}")
    
    # Save GSRSL
    # 保存GSRSL
    save_gsrsl(class_means, output_path)
    
    print(f"✓ GSRSL saved successfully")
    print(f"✓ GSRSL保存成功")
    print()
    
    # Load and verify
    # 加载并验证
    print("Loading saved GSRSL to verify...")
    print("加载保存的GSRSL以验证...")
    loaded = np.load(output_path, allow_pickle=True).item()
    
    print(f"✓ Loaded GSRSL with {len(loaded)} classes")
    print(f"✓ 加载了包含{len(loaded)}个类别的GSRSL")
    
    # Print statistics for each class
    # 打印每个类别的统计信息
    class_names = {
        0: "Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)",
        1: "Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)",
        2: "Mixed-type Rich Pegmatite (混合型富矿伟晶岩)",
        3: "Barren Pegmatite (贫矿伟晶岩)",
        4: "Wall Rock (围岩)"
    }
    
    print("\nClass statistics:")
    print("类别统计:")
    for class_id in range(5):
        spectrum = loaded[class_id]
        mean_refl = np.mean(spectrum)
        min_refl = np.min(spectrum)
        max_refl = np.max(spectrum)
        print(f"  Class {class_id} ({class_names[class_id]}):")
        print(f"    Mean reflectance: {mean_refl:.4f}")
        print(f"    Range: [{min_refl:.4f}, {max_refl:.4f}]")
    
    print()


def demo_visualize_gsrsl(class_means, wavelengths, output_dir='examples'):
    """
    Demonstrate visualizing GSRSL.
    演示可视化GSRSL
    
    Args:
        class_means: Dictionary of class mean spectra
                    类别平均光谱的字典
        wavelengths: Wavelength array
                    波长数组
        output_dir: Output directory
                   输出目录
    """
    print("=" * 70)
    print("Demo 2: Visualizing GSRSL")
    print("演示2: 可视化GSRSL")
    print("=" * 70)
    
    # Define output path
    # 定义输出路径
    output_path = os.path.join(output_dir, 'demo_gsrsl_visualization.png')
    
    print(f"Generating visualization: {output_path}")
    print(f"生成可视化: {output_path}")
    
    # Generate visualization
    # 生成可视化
    visualize_gsrsl(class_means, wavelengths, output_path)
    
    print(f"✓ Visualization saved successfully")
    print(f"✓ 可视化保存成功")
    print()
    
    # Display file size
    # 显示文件大小
    file_size = os.path.getsize(output_path)
    print(f"File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    print(f"文件大小: {file_size:,} 字节 ({file_size/1024:.2f} KB)")
    print()


def demo_with_nan_values(wavelengths, output_dir='examples'):
    """
    Demonstrate handling of NaN values.
    演示NaN值的处理
    
    Args:
        wavelengths: Wavelength array
                    波长数组
        output_dir: Output directory
                   输出目录
    """
    print("=" * 70)
    print("Demo 3: Handling NaN values")
    print("演示3: 处理NaN值")
    print("=" * 70)
    
    # Create GSRSL with some NaN values
    # 创建包含一些NaN值的GSRSL
    class_means_nan = {}
    for class_id in range(5):
        spectrum = np.full(297, 0.3 + class_id * 0.1, dtype=np.float32)
        # Set some bands to NaN (simulating no overlap)
        # 将一些波段设置为NaN(模拟无重叠)
        if class_id == 0:
            spectrum[0:5] = np.nan  # First 5 bands
        elif class_id == 2:
            spectrum[-10:] = np.nan  # Last 10 bands
        class_means_nan[class_id] = spectrum
    
    print("Created GSRSL with NaN values:")
    print("创建了包含NaN值的GSRSL:")
    for class_id in range(5):
        n_nan = np.sum(np.isnan(class_means_nan[class_id]))
        if n_nan > 0:
            print(f"  Class {class_id}: {n_nan} NaN bands")
    print()
    
    # Save GSRSL with NaN
    # 保存包含NaN的GSRSL
    output_path = os.path.join(output_dir, 'demo_gsrsl_with_nan.npy')
    print(f"Saving GSRSL with NaN to: {output_path}")
    print(f"保存包含NaN的GSRSL到: {output_path}")
    save_gsrsl(class_means_nan, output_path)
    print(f"✓ Saved successfully")
    print(f"✓ 保存成功")
    print()
    
    # Visualize GSRSL with NaN
    # 可视化包含NaN的GSRSL
    viz_path = os.path.join(output_dir, 'demo_gsrsl_with_nan_visualization.png')
    print(f"Generating visualization with NaN: {viz_path}")
    print(f"生成包含NaN的可视化: {viz_path}")
    visualize_gsrsl(class_means_nan, wavelengths, viz_path)
    print(f"✓ Visualization saved (NaN values shown as gaps)")
    print(f"✓ 可视化保存成功(NaN值显示为间隙)")
    print()


def main():
    """Main demo function"""
    print("\n" + "=" * 70)
    print("GSRSL Output Generator Module Demo")
    print("GSRSL输出生成器模块演示")
    print("=" * 70)
    print()
    
    # Setup logging
    # 设置日志
    setup_logging(log_file='examples/demo_output.log', level=20)  # INFO level
    
    # Create synthetic GSRSL data
    # 创建合成GSRSL数据
    class_means, wavelengths = create_synthetic_gsrsl()
    
    # Demo 1: Save GSRSL
    # 演示1: 保存GSRSL
    demo_save_gsrsl(class_means)
    
    # Demo 2: Visualize GSRSL
    # 演示2: 可视化GSRSL
    demo_visualize_gsrsl(class_means, wavelengths)
    
    # Demo 3: Handle NaN values
    # 演示3: 处理NaN值
    demo_with_nan_values(wavelengths)
    
    print("=" * 70)
    print("All demos completed successfully!")
    print("所有演示成功完成!")
    print("=" * 70)
    print()
    print("Output files created in 'examples/' directory:")
    print("在'examples/'目录中创建的输出文件:")
    print("  - demo_gsrsl.npy")
    print("  - demo_gsrsl_visualization.png")
    print("  - demo_gsrsl_with_nan.npy")
    print("  - demo_gsrsl_with_nan_visualization.png")
    print("  - demo_output.log")
    print()


if __name__ == '__main__':
    main()
