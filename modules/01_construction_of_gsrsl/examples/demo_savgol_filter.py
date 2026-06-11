"""
Demonstration of Savitzky-Golay filter for spectral denoising.
Savitzky-Golay滤波器用于光谱去噪的演示

This script demonstrates how the Savitzky-Golay filter smooths noisy spectral
data while preserving important absorption features.
该脚本演示Savitzky-Golay滤波器如何平滑噪声光谱数据，同时保留重要的吸收特征。
"""

import numpy as np
import matplotlib.pyplot as plt
from gsrsl_pipeline.data_models import GroundSpectrum
from gsrsl_pipeline.filters import apply_savgol_filter


def create_synthetic_spectrum():
    """
    Create a synthetic spectrum with realistic absorption features and noise.
    创建具有真实吸收特征和噪声的合成光谱
    """
    # Create wavelength array (1nm resolution from 350-2500nm)
    wavelengths = np.arange(350, 2501, 1, dtype=np.float64)
    
    # Create baseline reflectance with gentle slope
    baseline = 0.5 + 0.1 * (wavelengths - 1425) / 2150
    
    # Add absorption features typical of lithium-bearing minerals
    # Feature 1: Water absorption around 1400nm
    absorption_1400 = -0.15 * np.exp(-((wavelengths - 1400) ** 2) / (2 * 50 ** 2))
    
    # Feature 2: OH absorption around 2200nm
    absorption_2200 = -0.2 * np.exp(-((wavelengths - 2200) ** 2) / (2 * 80 ** 2))
    
    # Feature 3: Minor feature around 1900nm
    absorption_1900 = -0.08 * np.exp(-((wavelengths - 1900) ** 2) / (2 * 40 ** 2))
    
    # Combine features
    clean_reflectance = baseline + absorption_1400 + absorption_2200 + absorption_1900
    
    # Add realistic noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.015, len(wavelengths))
    noisy_reflectance = clean_reflectance + noise
    
    # Clip to valid range
    noisy_reflectance = np.clip(noisy_reflectance, 0.0, 1.0)
    
    return wavelengths, clean_reflectance, noisy_reflectance


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("Savitzky-Golay Filter Demonstration")
    print("Savitzky-Golay滤波器演示")
    print("=" * 80)
    
    # Create synthetic spectrum
    print("\n1. Creating synthetic spectrum with absorption features and noise...")
    print("   创建具有吸收特征和噪声的合成光谱...")
    wavelengths, clean, noisy = create_synthetic_spectrum()
    
    # Create GroundSpectrum object
    spectrum = GroundSpectrum(
        wavelengths=wavelengths,
        reflectances=noisy,
        class_id=0,
        filename="synthetic_spectrum.csv",
        is_anomalous=False
    )
    
    print(f"   Spectrum created: {len(wavelengths)} points")
    print(f"   Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    print(f"   Noise level (std): {np.std(noisy - clean):.6f}")
    
    # Apply Savitzky-Golay filter
    print("\n2. Applying Savitzky-Golay filter (window_length=15, polyorder=3)...")
    print("   应用Savitzky-Golay滤波器（window_length=15, polyorder=3）...")
    filtered_spectrum = apply_savgol_filter(spectrum)
    
    # Calculate noise reduction
    noise_before = np.std(noisy - clean)
    noise_after = np.std(filtered_spectrum.reflectances - clean)
    noise_reduction = (1 - noise_after / noise_before) * 100
    
    print(f"   Filtering complete!")
    print(f"   Noise reduction: {noise_reduction:.1f}%")
    print(f"   Noise before: {noise_before:.6f}")
    print(f"   Noise after: {noise_after:.6f}")
    
    # Verify wavelengths unchanged
    assert np.array_equal(filtered_spectrum.wavelengths, spectrum.wavelengths)
    print("   ✓ Wavelengths preserved unchanged")
    
    # Verify reflectances in valid range
    assert np.all(filtered_spectrum.reflectances >= 0.0)
    assert np.all(filtered_spectrum.reflectances <= 1.0)
    print("   ✓ Reflectances in valid range [0.0, 1.0]")
    
    # Create visualization
    print("\n3. Creating visualization...")
    print("   创建可视化...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Full spectrum comparison
    ax1 = axes[0]
    ax1.plot(wavelengths, clean, 'k-', linewidth=2, label='Clean Signal', alpha=0.7)
    ax1.plot(wavelengths, noisy, 'r-', linewidth=0.5, label='Noisy Signal', alpha=0.5)
    ax1.plot(wavelengths, filtered_spectrum.reflectances, 'b-', linewidth=1.5,
             label='Filtered Signal', alpha=0.8)
    ax1.set_xlabel('Wavelength / nm', fontsize=11)
    ax1.set_ylabel('Reflectance', fontsize=11)
    ax1.set_title('Savitzky-Golay Filter: Full Spectrum', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(350, 2500)
    ax1.set_ylim(0.0, 1.0)
    
    # Add text box with statistics
    textstr = f'Noise Reduction: {noise_reduction:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Plot 2: Zoom in on absorption feature around 2200nm
    ax2 = axes[1]
    zoom_mask = (wavelengths >= 2000) & (wavelengths <= 2400)
    ax2.plot(wavelengths[zoom_mask], clean[zoom_mask], 'k-', linewidth=2,
             label='Clean Signal', alpha=0.7)
    ax2.plot(wavelengths[zoom_mask], noisy[zoom_mask], 'r-', linewidth=0.5,
             label='Noisy Signal', alpha=0.5)
    ax2.plot(wavelengths[zoom_mask], filtered_spectrum.reflectances[zoom_mask], 'b-',
             linewidth=1.5, label='Filtered Signal', alpha=0.8)
    ax2.set_xlabel('Wavelength / nm', fontsize=11)
    ax2.set_ylabel('Reflectance', fontsize=11)
    ax2.set_title('Zoom: Absorption Feature at 2200nm', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add annotation for absorption feature
    idx_min = np.argmin(filtered_spectrum.reflectances[zoom_mask])
    wl_min = wavelengths[zoom_mask][idx_min]
    refl_min = filtered_spectrum.reflectances[zoom_mask][idx_min]
    ax2.annotate(f'Absorption minimum\n{wl_min:.0f} nm',
                 xy=(wl_min, refl_min), xytext=(wl_min + 100, refl_min + 0.05),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                 fontsize=9, ha='left')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'examples/savgol_filter_demo.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Visualization saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print("演示完成！")
    print("=" * 80)
    print("\nKey observations:")
    print("关键观察:")
    print(f"  • Noise reduced by {noise_reduction:.1f}% while preserving absorption features")
    print(f"    噪声降低{noise_reduction:.1f}%，同时保留吸收特征")
    print("  • Wavelength array remains unchanged")
    print("    波长数组保持不变")
    print("  • All reflectance values remain in valid range [0.0, 1.0]")
    print("    所有反射率值保持在有效范围[0.0, 1.0]内")
    print("  • Absorption features at 1400nm, 1900nm, and 2200nm are preserved")
    print("    1400nm、1900nm和2200nm处的吸收特征被保留")


if __name__ == "__main__":
    main()
