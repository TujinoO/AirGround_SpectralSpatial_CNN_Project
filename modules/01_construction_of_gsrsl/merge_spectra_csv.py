import sys
from pathlib import Path

import numpy as np
import pandas as pd


def merge_spectra(output_path: Path, *input_paths: str) -> None:
    """
    Merge multiple matrix-format spectra CSV files into a single spectra.csv.
    
    Expected input format for each file:
        - Column 0 header: 'Wavelength_(nm)' (or compatible wavelength column)
        - Column 0 values: wavelengths in nm
        - Columns 1..N: reflectance values for each sample spectrum
    
    All input files must share the same wavelength grid.
    """
    if len(input_paths) == 0:
        raise ValueError("At least one input spectra CSV file must be provided.")
    
    dfs = []
    wavelength_col_name = None
    wavelength_values = None
    
    for idx, path_str in enumerate(input_paths):
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Input spectra file not found: '{path}'")
        
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError(
                f"Spectrum matrix file '{path}' must have at least 2 columns. "
                f"Found {df.shape[1]} column(s). "
                f"Expected format: column 0 = wavelength (nm), columns 1..N = reflectance."
            )
        
        if idx == 0:
            wavelength_col_name = df.columns[0]
            wavelength_values = df.iloc[:, 0].to_numpy(dtype=np.float64)
        else:
            current_wavelengths = df.iloc[:, 0].to_numpy(dtype=np.float64)
            if not np.allclose(wavelength_values, current_wavelengths, rtol=0.0, atol=1e-8):
                raise ValueError(
                    f"Wavelength grid mismatch between files. "
                    f"First file uses column '{wavelength_col_name}' from '{input_paths[0]}', "
                    f"but file '{path}' has different wavelength values."
                )
        
        dfs.append(df)
    
    merged = pd.DataFrame()
    merged[wavelength_col_name] = wavelength_values
    
    for df in dfs:
        for col in df.columns[1:]:
            if col in merged.columns:
                raise ValueError(
                    f"Duplicate spectrum column name detected: '{col}'. "
                    f"Please ensure all spectrum column names are unique across input files."
                )
            merged[col] = df[col].to_numpy()
    
    output_path = output_path.resolve()
    merged.to_csv(output_path, index=False)


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("Usage:")
        print("  python merge_spectra_csv.py <output_csv> <input_csv_1> <input_csv_2> [<input_csv_3> ...]")
        return 1
    
    output = Path(argv[1])
    inputs = argv[2:]
    
    try:
        merge_spectra(output, *inputs)
    except Exception as e:
        print(f"Error during merge: {e}")
        return 1
    
    print(f"Merged spectra written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

