"""
Demo script for class aggregator module.
类别聚合器模块的演示脚本

This script demonstrates how to use the aggregate_by_class function to compute
mean reference spectra for each lithology class.
该脚本演示如何使用aggregate_by_class函数计算每个岩性类别的平均参考光谱。
"""

import numpy as np
import sys
import os

# Add parent directory to path to import gsrsl_pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gsrsl_pipeline.aggregator import aggregate_by_class
from gsrsl_pipeline.resampler import SpectralResampler
from gsrsl_pipeline.data_models import GroundSpectrum
from gsrsl_pipeline.filters import apply_savgol_filter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_spectrum(class_id: int, base_reflectance: float) -> GroundSpectrum:
    """
    Create a sample ground spectrum for demonstration.
    创建用于演示的样本地面光谱
    
    Args:
        class_id: Lithology class (0-4)
        base_reflectance: Base reflectance value
    
    Returns:
        GroundSpectrum object
    """
    # Create wavelength array from 350 to 2500 nm at 1nm resolution
    wavelengths = np.linspace(350, 2500, 2151, dtype=np.float64)
    
    # Create reflectance with some spectral features
    reflectances = base_reflectance + 0.1 * np.sin(wavelengths / 200.0)
    reflectances = np.clip(reflectances, 0.0, 1.0).astype(np.float64)
    
    return GroundSpectrum(
        wavelengths=wavelengths,
        reflectances=reflectances,
        class_id=class_id,
        filename=f"sample_class{class_id}.csv",
        is_anomalous=False
    )


def main():
    """Main demonstration function."""
    
    logger.info("=" * 70)
    logger.info("Class Aggregator Demo")
    logger.info("类别聚合器演示")
    logger.info("=" * 70)
    
    # Step 1: Create sample ground spectra for each class
    logger.info("\nStep 1: Creating sample ground spectra")
    logger.info("步骤1：创建样本地面光谱")
    
    ground_spectra = []
    
    # Class 0: Spodumene-rich Pegmatite (3 samples)
    ground_spectra.append(create_sample_spectrum(0, 0.25))
    ground_spectra.append(create_sample_spectrum(0, 0.30))
    ground_spectra.append(create_sample_spectrum(0, 0.35))
    
    # Class 1: Lepidolite-rich Pegmatite (2 samples)
    ground_spectra.append(create_sample_spectrum(1, 0.40))
    ground_spectra.append(create_sample_spectrum(1, 0.45))
    
    # Class 2: Mixed-type Rich Pegmatite (2 samples)
    ground_spectra.append(create_sample_spectrum(2, 0.50))
    ground_spectra.append(create_sample_spectrum(2, 0.55))
    
    # Class 3: Barren Pegmatite (1 sample)
    ground_spectra.append(create_sample_spectrum(3, 0.60))
    
    # Class 4: Wall Rock (2 samples)
    ground_spectra.append(create_sample_spectrum(4, 0.65))
    ground_spectra.append(create_sample_spectrum(4, 0.70))
    
    logger.info(f"Created {len(ground_spectra)} ground spectra")
    logger.info(f"创建了{len(ground_spectra)}个地面光谱")
    
    # Step 2: Apply Savitzky-Golay filter
    logger.info("\nStep 2: Applying Savitzky-Golay filter")
    logger.info("步骤2：应用Savitzky-Golay滤波器")
    
    filtered_spectra = [apply_savgol_filter(spectrum) for spectrum in ground_spectra]
    logger.info("Filtering complete")
    logger.info("滤波完成")
    
    # Step 3: Create satellite band specifications (simplified GF-5)
    logger.info("\nStep 3: Creating satellite band specifications")
    logger.info("步骤3：创建卫星波段规格")
    
    sat_wavelengths = np.linspace(450, 2450, 297, dtype=np.float64)
    sat_fwhms = np.linspace(4.0, 8.0, 297, dtype=np.float64)
    
    logger.info(f"Satellite bands: {len(sat_wavelengths)}")
    logger.info(f"卫星波段数：{len(sat_wavelengths)}")
    logger.info(f"Wavelength range: {sat_wavelengths[0]:.1f} - {sat_wavelengths[-1]:.1f} nm")
    logger.info(f"波长范围：{sat_wavelengths[0]:.1f} - {sat_wavelengths[-1]:.1f} nm")
    
    # Step 4: Resample spectra to satellite bands
    logger.info("\nStep 4: Resampling spectra to satellite bands")
    logger.info("步骤4：将光谱重采样到卫星波段")
    
    resampler = SpectralResampler(sat_wavelengths, sat_fwhms)
    
    resampled_spectra = []
    for spectrum in filtered_spectra:
        resampled = resampler.resample(spectrum)
        resampled_spectra.append((spectrum.class_id, resampled))
    
    logger.info(f"Resampled {len(resampled_spectra)} spectra")
    logger.info(f"重采样了{len(resampled_spectra)}个光谱")
    
    # Step 5: Aggregate by class
    logger.info("\nStep 5: Aggregating spectra by lithology class")
    logger.info("步骤5：按岩性类别聚合光谱")
    
    class_means = aggregate_by_class(resampled_spectra)
    
    # Step 6: Display results
    logger.info("\n" + "=" * 70)
    logger.info("Results / 结果")
    logger.info("=" * 70)
    
    class_names = {
        0: "Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)",
        1: "Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)",
        2: "Mixed-type Rich Pegmatite (混合型富矿伟晶岩)",
        3: "Barren Pegmatite (贫矿伟晶岩)",
        4: "Wall Rock (围岩)"
    }
    
    for class_id in range(5):
        mean_spectrum = class_means[class_id]
        
        # Count samples for this class
        n_samples = sum(1 for cid, _ in resampled_spectra if cid == class_id)
        
        # Calculate statistics
        valid_values = mean_spectrum[~np.isnan(mean_spectrum)]
        mean_reflectance = np.mean(valid_values)
        min_reflectance = np.min(valid_values)
        max_reflectance = np.max(valid_values)
        n_nan = np.sum(np.isnan(mean_spectrum))
        
        logger.info(f"\nClass {class_id}: {class_names[class_id]}")
        logger.info(f"  Samples: {n_samples}")
        logger.info(f"  样本数：{n_samples}")
        logger.info(f"  Mean reflectance: {mean_reflectance:.4f}")
        logger.info(f"  平均反射率：{mean_reflectance:.4f}")
        logger.info(f"  Reflectance range: [{min_reflectance:.4f}, {max_reflectance:.4f}]")
        logger.info(f"  反射率范围：[{min_reflectance:.4f}, {max_reflectance:.4f}]")
        logger.info(f"  NaN bands: {n_nan} / 297 ({n_nan/297*100:.1f}%)")
        logger.info(f"  NaN波段：{n_nan} / 297 ({n_nan/297*100:.1f}%)")
    
    logger.info("\n" + "=" * 70)
    logger.info("Demo complete!")
    logger.info("演示完成！")
    logger.info("=" * 70)
    
    return class_means


if __name__ == "__main__":
    class_means = main()
