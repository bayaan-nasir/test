"""
scripts/download_anaemia_dataset.py

Dataset:
  Kaggle: https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset
  ~1,400 rows | 5 features (Gender, Hemoglobin, MCH, MCHC, MCV) | Binary target

Expected Output: data\\raw\\anaemia\\anaemia.csv
"""
import os
from pathlib import Path

DATA_DIR       = Path("data/raw/anaemia")
KAGGLE_DATASET = "biswaranjanrao/anemia-dataset"


def download():
    print("Downloading Anaemia dataset from Kaggle...")
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
    target_csv = DATA_DIR / "anaemia.csv"
    if source_csv != target_csv:
        source_csv.rename(target_csv)
        print(f"Renamed {source_csv.name} -> anaemia.csv")

    df = pd.read_csv(target_csv)
    print(f"\nDataset loaded: {len(df)} rows x {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    if "Result" in df.columns:
        print(f"Class balance ('Result'):\n{df['Result'].value_counts()}")


if __name__ == "__main__":
    download()
    verify()
    print("\nDone. Run training with:")
    print("  python -m training.anaemia.train")
