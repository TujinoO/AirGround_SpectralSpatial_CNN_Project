"""
Integration tests for build_gsrsl.py main pipeline script
build_gsrsl.py主管道脚本的集成测试
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
from pathlib import Path
import subprocess
import sys


class TestBuildGSRSLIntegration:
    """Integration tests for the complete GSRSL pipeline"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace with test data"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create subdirectories
        metadata_dir = Path(temp_dir) / "metadata"
        output_dir = Path(temp_dir) / "output"
        
        metadata_dir.mkdir()
        output_dir.mkdir()
        
        # Create mock GF-5 metadata file
        metadata_path = metadata_dir / "gf5_metadata.txt"
        with open(metadata_path, 'w') as f:
            for i in range(1, 298):
                wavelength = 387.21 + (i - 1) * 7.0
                fwhm = 4.38 + (i - 1) * 0.01
                f.write(f"Wavelengths {i} = {wavelength:.2f}\n")
                f.write(f"FWHM {i} = {fwhm:.2f}\n")
        
        # Create mock ground spectra matrix file (5 samples, 1 per class)
        wavelengths = np.linspace(350, 2500, 2151, dtype=np.float64)
        spectra_df = pd.DataFrame()
        spectra_df["Wavelength_(nm)"] = wavelengths
        
        for class_id in range(5):
            base_reflectance = 0.2 + class_id * 0.1
            reflectances = np.full_like(wavelengths, base_reflectance)
            reflectances += 0.05 * np.sin(wavelengths / 200)
            reflectances = np.clip(reflectances, 0.0, 1.0)
            column_name = f"sample_class{class_id}.csv"
            spectra_df[column_name] = reflectances.astype(np.float64)
        
        spectra_path = Path(temp_dir) / "spectra_matrix.csv"
        spectra_df.to_csv(spectra_path, index=False)
        
        # Create label table
        labels_path = metadata_dir / "labels.csv"
        label_data = {
            'filename': [f"sample_class{i}.csv" for i in range(5)],
            'class_id': list(range(5))
        }
        pd.DataFrame(label_data).to_csv(labels_path, index=False)
        
        # Return paths
        workspace = {
            'root': temp_dir,
            'metadata': str(metadata_path),
            'spectra': str(spectra_path),
            'labels': str(labels_path),
            'output': str(output_dir)
        }
        
        yield workspace
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_pipeline_help_message(self):
        """Test that the pipeline script shows help message"""
        result = subprocess.run(
            [sys.executable, 'build_gsrsl.py', '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Build Ground Standard Reference Spectral Library' in result.stdout
        assert '--metadata' in result.stdout
        assert '--spectra' in result.stdout
        assert '--labels' in result.stdout
        assert '--output' in result.stdout
    
    def test_pipeline_missing_arguments(self):
        """Test that the pipeline fails gracefully with missing arguments"""
        result = subprocess.run(
            [sys.executable, 'build_gsrsl.py'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0
        assert 'required' in result.stderr.lower() or 'required' in result.stdout.lower()
    
    def test_pipeline_end_to_end(self, temp_workspace):
        """Test complete pipeline execution with synthetic data"""
        # Run the pipeline
        result = subprocess.run(
            [
                sys.executable, 'build_gsrsl.py',
                '--metadata', temp_workspace['metadata'],
                '--spectra', temp_workspace['spectra'],
                '--labels', temp_workspace['labels'],
                '--output', temp_workspace['output'],
                '--log', os.path.join(temp_workspace['output'], 'test.log')
            ],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        # Check exit code
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, f"Pipeline failed with exit code {result.returncode}"
        
        # Verify output files exist
        gsrsl_path = Path(temp_workspace['output']) / 'gsrsl.npy'
        viz_path = Path(temp_workspace['output']) / 'gsrsl_visualization.png'
        log_path = Path(temp_workspace['output']) / 'test.log'
        
        assert gsrsl_path.exists(), "GSRSL file not created"
        assert viz_path.exists(), "Visualization file not created"
        assert log_path.exists(), "Log file not created"
        
        # Verify GSRSL content
        gsrsl = np.load(gsrsl_path, allow_pickle=True).item()
        
        assert isinstance(gsrsl, dict), "GSRSL should be a dictionary"
        assert set(gsrsl.keys()) == {0, 1, 2, 3, 4}, "GSRSL should have 5 classes"
        
        for class_id in range(5):
            spectrum = gsrsl[class_id]
            assert isinstance(spectrum, np.ndarray), f"Class {class_id} should be numpy array"
            assert spectrum.shape == (297,), f"Class {class_id} should have 297 bands"
            assert spectrum.dtype == np.float32, f"Class {class_id} should be float32"
            
            # Check that most values are valid (not NaN)
            valid_count = np.sum(~np.isnan(spectrum))
            assert valid_count > 200, f"Class {class_id} should have mostly valid bands"
            
            # Check reflectance range for valid values
            valid_values = spectrum[~np.isnan(spectrum)]
            assert np.all(valid_values >= 0.0), f"Class {class_id} has negative reflectances"
            assert np.all(valid_values <= 1.0), f"Class {class_id} has reflectances > 1.0"
        
        # Verify log file contains expected messages
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        assert 'GSRSL Pipeline Started' in log_content
        assert 'Successfully parsed metadata' in log_content
        assert 'Processing Ground Spectra' in log_content
        assert 'Aggregating Spectra by Lithology Class' in log_content
        assert 'GSRSL Pipeline Completed Successfully' in log_content
    
    def test_pipeline_with_invalid_metadata_path(self, temp_workspace):
        """Test that pipeline fails gracefully with invalid metadata path"""
        result = subprocess.run(
            [
                sys.executable, 'build_gsrsl.py',
                '--metadata', 'nonexistent_metadata.txt',
                '--spectra', temp_workspace['spectra'],
                '--labels', temp_workspace['labels'],
                '--output', temp_workspace['output']
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode != 0
        # Check that error message is informative
        output = result.stdout + result.stderr
        assert 'not found' in output.lower() or 'metadata' in output.lower()
    
    def test_pipeline_with_invalid_spectra_directory(self, temp_workspace):
        """Test that pipeline fails gracefully with invalid spectra directory"""
        result = subprocess.run(
            [
                sys.executable, 'build_gsrsl.py',
                '--metadata', temp_workspace['metadata'],
                '--spectra', 'nonexistent_directory',
                '--labels', temp_workspace['labels'],
                '--output', temp_workspace['output']
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert 'not found' in output.lower() or 'directory' in output.lower()
    
    def test_pipeline_with_verbose_flag(self, temp_workspace):
        """Test that verbose flag enables detailed logging"""
        result = subprocess.run(
            [
                sys.executable, 'build_gsrsl.py',
                '--metadata', temp_workspace['metadata'],
                '--spectra', temp_workspace['spectra'],
                '--labels', temp_workspace['labels'],
                '--output', temp_workspace['output'],
                '--verbose'
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert result.returncode == 0
        # Verbose mode should produce more output
        assert len(result.stdout) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
