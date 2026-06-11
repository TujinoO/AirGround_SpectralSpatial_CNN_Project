"""
Demo script for build_gsrsl.py main pipeline
build_gsrsl.py主管道的演示脚本

This script demonstrates how to use the complete GSRSL pipeline with synthetic data.
该脚本演示如何使用完整的GSRSL管道处理合成数据。
"""

import numpy as np
import pandas as pd
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path


def create_demo_data():
    """
    Create synthetic demo data for the GSRSL pipeline.
    为GSRSL管道创建合成演示数据
    
    Returns:
        Dictionary with paths to demo data files
        包含演示数据文件路径的字典
    """
    print("Creating synthetic demo data...")
    print("创建合成演示数据...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix='gsrsl_demo_')
    print(f"  Demo workspace: {temp_dir}")
    
    # Create subdirectories
    metadata_dir = Path(temp_dir) / "metadata"
    spectra_dir = Path(temp_dir) / "spectra"
    output_dir = Path(temp_dir) / "output"
    
    metadata_dir.mkdir()
    spectra_dir.mkdir()
    output_dir.mkdir()
    
    # 1. Create mock GF-5 metadata file
    print("\n  Creating GF-5 metadata file...")
    metadata_path = metadata_dir / "gf5_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("# GF-5 Satellite Metadata (Synthetic)\n")
        f.write("# 297 spectral bands\n\n")
        for i in range(1, 298):
            wavelength = 387.21 + (i - 1) * 7.0  # ~387-2465 nm
            fwhm = 4.38 + (i - 1) * 0.01  # ~4.38-7.34 nm
            f.write(f"Wavelengths {i} = {wavelength:.2f}\n")
            f.write(f"FWHM {i} = {fwhm:.2f}\n")
    print(f"    Created: {metadata_path}")
    
    # 2. Create mock ground spectra files
    print("\n  Creating ground spectrum files...")
    wavelengths = np.linspace(350, 2500, 2151, dtype=np.float64)
    
    # Class names for reference
    class_names = {
        0: "Spodumene-rich Pegmatite (锂辉石型富矿伟晶岩)",
        1: "Lepidolite-rich Pegmatite (锂云母型富矿伟晶岩)",
        2: "Mixed-type Rich Pegmatite (混合型富矿伟晶岩)",
        3: "Barren Pegmatite (贫矿伟晶岩)",
        4: "Wall Rock (围岩)"
    }
    
    # Create 3 samples per class (15 total)
    filenames = []
    class_ids = []
    
    for class_id in range(5):
        print(f"    Class {class_id}: {class_names[class_id]}")
        
        for sample_num in range(3):
            # Create unique spectrum for each sample
            base_reflectance = 0.15 + class_id * 0.12
            reflectances = np.full_like(wavelengths, base_reflectance)
            
            # Add class-specific absorption features
            if class_id == 0:  # Spodumene-rich
                # Strong absorption at 2200nm (Al-OH)
                reflectances -= 0.15 * np.exp(-((wavelengths - 2200) ** 2) / (2 * 40 ** 2))
            elif class_id == 1:  # Lepidolite-rich
                # Absorption at 2200nm and 1400nm
                reflectances -= 0.12 * np.exp(-((wavelengths - 2200) ** 2) / (2 * 40 ** 2))
                reflectances -= 0.10 * np.exp(-((wavelengths - 1400) ** 2) / (2 * 50 ** 2))
            elif class_id == 2:  # Mixed-type
                # Moderate absorption at multiple wavelengths
                reflectances -= 0.08 * np.exp(-((wavelengths - 2200) ** 2) / (2 * 40 ** 2))
                reflectances -= 0.06 * np.exp(-((wavelengths - 1400) ** 2) / (2 * 50 ** 2))
            elif class_id == 3:  # Barren
                # Weak absorption features
                reflectances -= 0.05 * np.exp(-((wavelengths - 2200) ** 2) / (2 * 40 ** 2))
            else:  # Wall Rock
                # Iron oxide absorption around 900nm
                reflectances -= 0.10 * np.exp(-((wavelengths - 900) ** 2) / (2 * 30 ** 2))
            
            # Add sample-specific variation
            reflectances += 0.02 * np.sin(wavelengths / 200 + sample_num)
            
            # Add small random noise
            np.random.seed(class_id * 100 + sample_num)
            reflectances += np.random.normal(0, 0.005, len(wavelengths))
            
            # Clip to valid range
            reflectances = np.clip(reflectances, 0.0, 1.0)
            
            # Save to CSV (without header to match TSG8 format)
            filename = f"sample_class{class_id}_num{sample_num + 1}.csv"
            filepath = spectra_dir / filename
            
            data = np.column_stack([wavelengths, reflectances])
            np.savetxt(filepath, data, delimiter=',', fmt='%.6f')
            
            filenames.append(filename)
            class_ids.append(class_id)
            
            print(f"      Sample {sample_num + 1}: {filename}")
    
    # 3. Create label table
    print("\n  Creating label table...")
    labels_path = metadata_dir / "labels.csv"
    label_data = {
        'filename': filenames,
        'class_id': class_ids
    }
    pd.DataFrame(label_data).to_csv(labels_path, index=False)
    print(f"    Created: {labels_path}")
    print(f"    Total samples: {len(filenames)}")
    
    # Return paths
    return {
        'root': temp_dir,
        'metadata': str(metadata_path),
        'spectra': str(spectra_dir),
        'labels': str(labels_path),
        'output': str(output_dir)
    }


def run_pipeline(paths):
    """
    Run the GSRSL pipeline with demo data.
    使用演示数据运行GSRSL管道
    
    Args:
        paths: Dictionary with paths to data files
              包含数据文件路径的字典
    """
    print("\n" + "=" * 70)
    print("Running GSRSL Pipeline")
    print("运行GSRSL管道")
    print("=" * 70)
    
    # Build command
    cmd = [
        sys.executable, 'build_gsrsl.py',
        '--metadata', paths['metadata'],
        '--spectra', paths['spectra'],
        '--labels', paths['labels'],
        '--output', paths['output'],
        '--log', str(Path(paths['output']) / 'demo.log'),
        '--verbose'
    ]
    
    print("\nCommand:")
    print(" ".join(cmd))
    print()
    
    # Run pipeline
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print(result.stdout)
    
    if result.returncode != 0:
        print("\nERROR OUTPUT:")
        print(result.stderr)
        raise RuntimeError(f"Pipeline failed with exit code {result.returncode}")
    
    return result.returncode == 0


def verify_outputs(paths):
    """
    Verify that pipeline outputs were created correctly.
    验证管道输出是否正确创建
    
    Args:
        paths: Dictionary with paths to data files
              包含数据文件路径的字典
    """
    print("\n" + "=" * 70)
    print("Verifying Outputs")
    print("验证输出")
    print("=" * 70)
    
    output_dir = Path(paths['output'])
    
    # Check GSRSL file
    gsrsl_path = output_dir / 'gsrsl.npy'
    if gsrsl_path.exists():
        print(f"\n✓ GSRSL file created: {gsrsl_path}")
        
        # Load and inspect
        gsrsl = np.load(gsrsl_path, allow_pickle=True).item()
        print(f"  Classes: {list(gsrsl.keys())}")
        
        for class_id in range(5):
            spectrum = gsrsl[class_id]
            valid_bands = np.sum(~np.isnan(spectrum))
            mean_reflectance = np.nanmean(spectrum)
            print(f"  Class {class_id}: {valid_bands}/297 valid bands, "
                  f"mean reflectance = {mean_reflectance:.4f}")
    else:
        print(f"\n✗ GSRSL file not found: {gsrsl_path}")
    
    # Check visualization
    viz_path = output_dir / 'gsrsl_visualization.png'
    if viz_path.exists():
        print(f"\n✓ Visualization created: {viz_path}")
        file_size = viz_path.stat().st_size
        print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    else:
        print(f"\n✗ Visualization not found: {viz_path}")
    
    # Check log file
    log_path = output_dir / 'demo.log'
    if log_path.exists():
        print(f"\n✓ Log file created: {log_path}")
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"  Log entries: {len(lines)} lines")
    else:
        print(f"\n✗ Log file not found: {log_path}")


def main():
    """
    Main demo function.
    主演示函数
    """
    print("=" * 70)
    print("GSRSL Pipeline Demo")
    print("GSRSL管道演示")
    print("=" * 70)
    print()
    print("This demo creates synthetic ground spectral data and processes it")
    print("through the complete GSRSL pipeline to generate a reference library.")
    print("该演示创建合成地面光谱数据，并通过完整的GSRSL管道处理它以生成参考库。")
    print()
    
    try:
        # Step 1: Create demo data
        paths = create_demo_data()
        
        # Step 2: Run pipeline
        success = run_pipeline(paths)
        
        if success:
            # Step 3: Verify outputs
            verify_outputs(paths)
            
            print("\n" + "=" * 70)
            print("Demo Completed Successfully!")
            print("演示成功完成！")
            print("=" * 70)
            print(f"\nDemo outputs are in: {paths['output']}")
            print(f"演示输出位于：{paths['output']}")
            print("\nYou can inspect the following files:")
            print("您可以检查以下文件：")
            print(f"  - GSRSL data: {Path(paths['output']) / 'gsrsl.npy'}")
            print(f"  - Visualization: {Path(paths['output']) / 'gsrsl_visualization.png'}")
            print(f"  - Log file: {Path(paths['output']) / 'demo.log'}")
            
            # Ask if user wants to keep the demo data
            print("\nNote: Demo data is in a temporary directory.")
            print("注意：演示数据位于临时目录中。")
            print(f"Location: {paths['root']}")
            print("\nThe temporary directory will be deleted when you close this program.")
            print("当您关闭此程序时，临时目录将被删除。")
            
            input("\nPress Enter to cleanup and exit...")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if 'paths' in locals():
            print("\nCleaning up temporary files...")
            print("清理临时文件...")
            try:
                shutil.rmtree(paths['root'])
                print("✓ Cleanup complete")
            except Exception as e:
                print(f"✗ Cleanup failed: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
