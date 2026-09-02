"""
scripts/download_hepatitis_b_dataset.py
Downloads the UCI Hepatitis dataset from Kaggle.

Dataset:
  Kaggle: https://www.kaggle.com/datasets/codebreaker619/hepatitis-data
  155 rows | 19 original columns (11 used in our model) | Binary target
  Note: this is a small, real-world clinical dataset with many missing
  values (marked '?' in the raw UCI format) — the preprocessing pipeline
  in tabular_transforms.py handles this via imputation.

Output:
  data\\raw\\hepatitis_b\\hepatitis.csv
"""
import os
from pathlib import Path

DATA_DIR       = Path("data/raw/hepatitis_b")
KAGGLE_DATASET = "codebreaker619/hepatitis-data"


def download():
    print("Downloading Hepatitis B (UCI) dataset from Kaggle...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_DIR} --unzip")
    print(f"Download complete. Files saved to: {DATA_DIR}")


def verify():
    import pandas as pd

    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV found in {DATA_DIR}. Check your Kaggle credentials.")
        return

    source_csv = csv_files[0]
    target_csv = DATA_DIR / "hepatitis.csv"
    if source_csv != target_csv:
        source_csv.rename(target_csv)
        print(f"Renamed {source_csv.name} -> hepatitis.csv")

    df = pd.read_csv(target_csv)
    print(f"\nDataset loaded: {len(df)} rows x {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(
        "\nNOTE: this dataset is small (155 rows). Expect wider confidence "
        "intervals on test metrics than larger datasets in this project — "
        "this is worth mentioning explicitly in your thesis limitations section."
    )


if __name__ == "__main__":
    download()
    verify()
    print("\nDone. Run training with:")
    print("  python -m training.hepatitis_b.train")
