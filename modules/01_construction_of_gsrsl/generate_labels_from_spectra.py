import sys
from pathlib import Path
import csv
import pandas as pd


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: python generate_labels_from_spectra.py <spectra_csv> <labels_csv>")
        return 1
    spectra_path = Path(argv[1])
    labels_path = Path(argv[2])
    if not spectra_path.exists():
        print(f"Error: spectra file not found: {spectra_path}")
        return 1
    with spectra_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"Error: spectra file is empty: {spectra_path}")
            return 1
    if len(header) < 2:
        print("Error: spectra file must have at least two columns")
        return 1
    filenames = header[1:]
    existing_map = {}
    if labels_path.exists():
        try:
            existing = pd.read_csv(labels_path)
            if "filename" in existing.columns and "class_id" in existing.columns:
                existing_map = (
                    existing[["filename", "class_id"]]
                    .dropna(subset=["filename"])
                    .set_index("filename")["class_id"]
                    .to_dict()
                )
        except Exception:
            existing_map = {}
    rows = []
    for name in filenames:
        class_id = existing_map.get(name, "")
        rows.append({"filename": name, "class_id": class_id})
    df = pd.DataFrame(rows, columns=["filename", "class_id"])
    df.to_csv(labels_path, index=False)
    print(f"Labels written to: {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

