"""
scripts/download_hypertension_dataset.py
Downloads a Kaggle lifestyle-based hypertension risk dataset.

Dataset:
  Kaggle: https://www.kaggle.com/datasets/prosperchuks/health-dataset
  (or an equivalent hypertension risk dataset combining demographic,
  lifestyle, and basic clinical features)

NOTE: Kaggle hosts several similarly-named hypertension datasets with
slightly different column names. If the default dataset slug below
doesn't match what you find when browsing Kaggle, update KAGGLE_DATASET
below — the preprocessing.tabular_transforms.load_hypertension_dataframe()
function already handles several common column-naming variants.

Output:
  data\\raw\\hypertension\\hypertension.csv
"""
import os
from pathlib import Path

DATA_DIR       = Path("data/raw/hypertension")
KAGGLE_DATASET = "prosperchuks/health-dataset"


def download():
    print("Downloading Hypertension dataset from Kaggle...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_DIR} --unzip")
    print(f"Download complete. Files saved to: {DATA_DIR}")


def verify():
    import pandas as pd

    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV found in {DATA_DIR}. Check your Kaggle credentials, ")
        print("or verify the KAGGLE_DATASET slug in this script matches an actual Kaggle dataset.")
        return

    source_csv = csv_files[0]
    target_csv = DATA_DIR / "hypertension.csv"
    if source_csv != target_csv:
        source_csv.rename(target_csv)
        print(f"Renamed {source_csv.name} -> hypertension.csv")

    df = pd.read_csv(target_csv)
    print(f"\nDataset loaded: {len(df)} rows x {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(
        "\nIMPORTANT: Check that your CSV's column names map correctly in "
        "preprocessing/tabular_transforms.py -> load_hypertension_dataframe(). "
        "If your downloaded dataset uses different column names than expected "
        "(age, salt_intake, bmi, stress_score, sleep_hours, smoking, "
        "family_history, physical_activity, target), update the rename_map "
        "dictionary in that function."
    )


if __name__ == "__main__":
    download()
    verify()
    print("\nDone. Run training with:")
    print("  python -m training.hypertension.train")
