"""
scripts/download_skin_cancer_dataset.py
Downloads the HAM10000 skin lesion dataset from Kaggle and creates
stratified train/val/test split CSVs (does not copy/duplicate images —
just assigns each image_id to a split, since HAM10000 images stay in
their original two-part folder structure).

Dataset:
  Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
  10,015 images | 7 classes | HAM10000_metadata.csv has the labels

Output:
  data\\raw\\skin_cancer\\
    HAM10000_images_part_1\\*.jpg
    HAM10000_images_part_2\\*.jpg
    HAM10000_metadata.csv
    train_split.csv   (70%)
    val_split.csv      (15%)
    test_split.csv     (15%)

Stratified by 'dx' (diagnosis) to preserve class proportions across splits,
since some classes (df, vasc) have very few examples.
"""
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR       = Path("data/raw/skin_cancer")
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"


def download():
    print("Downloading HAM10000 dataset from Kaggle...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_DIR} --unzip")
    print(f"Download complete. Files saved to: {DATA_DIR}")


def create_splits():
    metadata_path = DATA_DIR / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} not found. Check the download.")
        return

    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata: {len(df)} rows")
    print(f"Class distribution:\n{df['dx'].value_counts()}")

    # Stratified 70/15/15 split, preserving class ratios
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["dx"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["dx"], random_state=42
    )

    train_df.to_csv(DATA_DIR / "train_split.csv", index=False)
    val_df.to_csv(DATA_DIR / "val_split.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_split.csv", index=False)

    print(f"\nSplit sizes: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")
    print("Split CSVs saved.")


def verify_image_folders():
    part1 = DATA_DIR / "HAM10000_images_part_1"
    part2 = DATA_DIR / "HAM10000_images_part_2"
    n1 = len(list(part1.glob("*.jpg"))) if part1.exists() else 0
    n2 = len(list(part2.glob("*.jpg"))) if part2.exists() else 0
    print(f"\nImage folders: part_1={n1} images | part_2={n2} images | total={n1 + n2}")
    if n1 + n2 == 0:
        print("WARNING: No images found. Check that both image part folders extracted correctly.")


if __name__ == "__main__":
    download()
    create_splits()
    verify_image_folders()
    print("\nDone. Run training with:")
    print("  python -m training.skin_cancer.train")
