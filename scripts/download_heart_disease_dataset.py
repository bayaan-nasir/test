"""
scripts/download_heart_disease_dataset.py
Downloads the UCI Heart Disease (Cleveland) dataset from Kaggle.

Dataset:
  Kaggle: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
  303 rows | 13 features | target column 0-4 (collapsed to binary during preprocessing)

Output:
  data\\raw\\heart_disease\\heart.csv
"""
import os
from pathlib import Path

DATA_DIR       = Path("data/raw/heart_disease")
KAGGLE_DATASET = "redwankarimsony/heart-disease-data"


def download():
    print("Downloading UCI Heart Disease dataset from Kaggle...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_DIR} --unzip")
    print(f"Download complete. Files saved to: {DATA_DIR}")


def verify():
    """
    The Kaggle download may name the CSV differently (e.g. heart_disease_uci.csv).
    This renames it to the expected heart.csv if needed.
    """
    import pandas as pd

    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV found in {DATA_DIR}. Check your Kaggle credentials.")
        return

    source_csv = csv_files[0]
    target_csv = DATA_DIR / "heart.csv"
    if source_csv != target_csv:
        source_csv.rename(target_csv)
        print(f"Renamed {source_csv.name} -> heart.csv")

    df = pd.read_csv(target_csv)
    print(f"\nDataset loaded: {len(df)} rows x {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    target_col = "target" if "target" in df.columns else "num"
    print(f"Target column '{target_col}' distribution:\n{df[target_col].value_counts()}")


if __name__ == "__main__":
    download()
    verify()
    print("\nDone. Run training with:")
    print("  python -m training.heart_disease.train")
