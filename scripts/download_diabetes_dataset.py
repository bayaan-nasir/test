# Script to download the PIMA Diabetes dataset from Kaggle and save it to the data/raw/diabetes directory.
import os
from pathlib import Path

DATA_DIR = Path("data/raw/diabetes")
KAGGLE_DATASET = "uciml/pima-indians-diabetes-database"


def download():
    print("Downloading PIMA Diabetes dataset from Kaggle...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.system(
        f"kaggle datasets download -d {KAGGLE_DATASET} "
        f"-p {DATA_DIR} --unzip"
    )
    print(f"Download complete. File saved to: {DATA_DIR}")


def verify():
    csv_path = DATA_DIR / "diabetes.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"\nDataset loaded: {len(df)} rows x {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"Class balance:\n{df['Outcome'].value_counts()}")
        print(f"\n0 = Non-Diabetic | 1 = Diabetic")
    else:
        print(f"ERROR: Expected file not found at {csv_path}")
        print("Check your Kaggle credentials and try again.")


if __name__ == "__main__":
    download()
    verify()
    print("\nDone. Run training with:")
    print("  python -m training.diabetes.train")
