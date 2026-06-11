"""
Demo script for SpectralResampler class.
SpectralResampler类的演示脚本

This script demonstrates how to use the SpectralResampler to convert
high-resolution ground spectra to satellite band resolution.
该脚本演示如何使用SpectralResampler将高分辨率地面光谱转换为卫星波段分辨率。
"""

import numpy as np
import matplotlib.pyplot as plt
from gsrsl_pipeline.resampler import SpectralResampler
from gsrsl_pipeline.data_models import GroundSpectrum


def create_synthetic_ground_spectrum():
    """
    Create a synthetic ground spectrum for demonstration.
    创建用于演示的合成地面光谱
    
    Returns:
        GroundSpectrum with realistic spectral features
        具有真实光谱特征的GroundSpectrum
    """
    # High-resolution ground spectrum: 1nm resolution, 350-2500nm range
    # 高分辨率地面光谱：1nm分辨率，350-2500nm范围
    wavelengths = np.linspace(350, 2500, 2151, dtype=np.float64)
    
    # Simulate a mineral spectrum with absorption features
    # 模拟具有吸收特征的矿物光谱
    # Base reflectance with gentle slope
    # 具有缓坡的基础反射率
    reflectances = 0.3 + 0.0001 * (wavelengths - 350)
    
    # Add absorption features (Gaussian dips)
    # 添加吸收特征（高斯凹陷）
    # Feature 1: Strong absorption around 1400nm (water)
    # 特征1：1400nm附近的强吸收（水）
    reflectances -= 0.15 * np.exp(-((wavelengths - 1400) ** 2) / (2 * 50 ** 2))
    
    # Feature 2: Moderate absorption around 2200nm (clay minerals)
    # 特征2：2200nm附近的中等吸收（粘土矿物）
    reflectances -= 0.10 * np.exp(-((wavelengths - 2200) ** 2) / (2 * 40 ** 2))
    
    # Feature 3: Weak absorption around 900nm (iron oxides)
    # 特征3：900nm附近的弱吸收（铁氧化物）
    reflectances -= 0.05 * np.exp(-((wavelengths - 900) ** 2) / (2 * 30 ** 2))
    
    # Clip to valid range [0, 1]
    # 裁剪到有效范围[0, 1]
    reflectances = np.clip(reflectances, 0.0, 1.0)
    
    # Create GroundSpectrum object
    # 创建GroundSpectrum对象
    spectrum = GroundSpectrum(
        wavelengths=wavelengths,
        reflectances=reflectances,
        class_id=0,
        filename="synthetic_mineral.csv",
        is_anomalous=False
    )
    
    return spectrum


def create_satellite_specifications():
    """
    Create simplified GF-5 satellite specifications.
    创建简化的GF-5卫星规格
    
    Returns:
        Tuple of (wavelengths, fwhms) for 297 bands
        297个波段的(wavelengths, fwhms)元组
    """
    # GF-5 covers approximately 450-2450nm with 297 bands
    # GF-5覆盖约450-2450nm，共297个波段
    sat_wavelengths = np.linspace(450, 2450, 297, dtype=np.float64)
    
    # FWHM varies from ~4nm in visible to ~8nm in SWIR
    # FWHM从可见光的约4nm到短波红外的约8nm变化
    sat_fwhms = np.linspace(4.0, 8.0, 297, dtype=np.float64)
    
    return sat_wavelengths, sat_fwhms


def main():
    """
    Main demonstration function.
    主演示函数
    """
    print("=" * 70)
    print("SpectralResampler Demonstration")
    print("SpectralResampler演示")
    print("=" * 70)
    print()
    
    # Step 1: Create synthetic ground spectrum
    # 步骤1：创建合成地面光谱
    print("Step 1: Creating synthetic ground spectrum...")
    print("步骤1：创建合成地面光谱...")
    ground_spectrum = create_synthetic_ground_spectrum()
    print(f"  Ground spectrum: {len(ground_spectrum)} points")
    print(f"  地面光谱：{len(ground_spectrum)}个点")
    print(f"  Wavelength range: [{ground_spectrum.wavelengths[0]:.1f}, "
          f"{ground_spectrum.wavelengths[-1]:.1f}] nm")
    print(f"  波长范围：[{ground_spectrum.wavelengths[0]:.1f}, "
          f"{ground_spectrum.wavelengths[-1]:.1f}] nm")
    print()
    
    # Step 2: Create satellite specifications
    # 步骤2：创建卫星规格
    print("Step 2: Creating satellite specifications...")
    print("步骤2：创建卫星规格...")
    sat_wavelengths, sat_fwhms = create_satellite_specifications()
    print(f"  Satellite bands: {len(sat_wavelengths)}")
    print(f"  卫星波段：{len(sat_wavelengths)}")
    print(f"  Wavelength range: [{sat_wavelengths[0]:.1f}, {sat_wavelengths[-1]:.1f}] nm")
    print(f"  波长范围：[{sat_wavelengths[0]:.1f}, {sat_wavelengths[-1]:.1f}] nm")
    print(f"  FWHM range: [{sat_fwhms[0]:.1f}, {sat_fwhms[-1]:.1f}] nm")
    print(f"  FWHM范围：[{sat_fwhms[0]:.1f}, {sat_fwhms[-1]:.1f}] nm")
    print()
    
    # Step 3: Create resampler
    # 步骤3：创建重采样器
    print("Step 3: Initializing SpectralResampler...")
    print("步骤3：初始化SpectralResampler...")
    resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
    print(f"  Computed sigmas range: [{resampler.sat_sigmas[0]:.2f}, "
          f"{resampler.sat_sigmas[-1]:.2f}] nm")
    print(f"  计算的sigma范围：[{resampler.sat_sigmas[0]:.2f}, "
          f"{resampler.sat_sigmas[-1]:.2f}] nm")
    print()
    
    # Step 4: Perform resampling
    # 步骤4：执行重采样
    print("Step 4: Resampling ground spectrum to satellite bands...")
    print("步骤4：将地面光谱重采样到卫星波段...")
    resampled = resampler.resample(ground_spectrum)
    print(f"  Resampled spectrum: {resampled.shape}")
    print(f"  重采样光谱：{resampled.shape}")
    print(f"  Valid bands: {np.sum(~np.isnan(resampled))} / {len(resampled)}")
    print(f"  有效波段：{np.sum(~np.isnan(resampled))} / {len(resampled)}")
    print(f"  Reflectance range: [{np.nanmin(resampled):.3f}, {np.nanmax(resampled):.3f}]")
    print(f"  反射率范围：[{np.nanmin(resampled):.3f}, {np.nanmax(resampled):.3f}]")
    print()
    
    # Step 5: Visualize results
    # 步骤5：可视化结果
    print("Step 5: Creating visualization...")
    print("步骤5：创建可视化...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Ground spectrum (high resolution)
    # 图1：地面光谱（高分辨率）
    ax1.plot(ground_spectrum.wavelengths, ground_spectrum.reflectances,
             'b-', linewidth=0.5, label='Ground Spectrum (1nm resolution)')
    ax1.set_xlabel('Wavelength / nm', fontsize=11)
    ax1.set_ylabel('Reflectance', fontsize=11)
    ax1.set_title('High-Resolution Ground Spectrum', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim(350, 2500)
    ax1.set_ylim(0, 0.6)
    
    # Plot 2: Comparison of ground and resampled spectra
    # 图2：地面光谱和重采样光谱的比较
    ax2.plot(ground_spectrum.wavelengths, ground_spectrum.reflectances,
             'b-', linewidth=0.5, alpha=0.5, label='Ground Spectrum (1nm)')
    ax2.plot(sat_wavelengths, resampled,
             'ro-', markersize=3, linewidth=1, label='Resampled to Satellite Bands')
    ax2.set_xlabel('Wavelength / nm', fontsize=11)
    ax2.set_ylabel('Reflectance', fontsize=11)
    ax2.set_title('Comparison: Ground VS Resampled', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim(350, 2500)
    ax2.set_ylim(0, 0.6)
    
    plt.tight_layout()
    
    # Save figure
    # 保存图形
    output_path = 'examples/resampler_demo_output.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Visualization saved to: {output_path}")
    print(f"  可视化已保存到：{output_path}")
    print()
    
    # Show plot
    # 显示图形
    plt.show()
    
    print("=" * 70)
    print("Demonstration complete!")
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
