# Script to download and prepare the Chest X-Ray Pneumonia dataset from Kaggle.
# The dataset is reorganised into data/raw/pneumonia/ with train/val/test splits.
import os
import shutil
import zipfile
from pathlib import Path
import random

DATA_DIR = Path("data/raw/pneumonia")
KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
ZIP_NAME = "chest-xray-pneumonia.zip"


def download():
    print("Downloading Chest X-Ray dataset from Kaggle...")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p data/raw/ --unzip")
    print("Download complete.")


def reorganise():
    """
    The Kaggle dataset extracts to chest_xray/. 
    Rename and restructure to data/raw/pneumonia/.
    Also fix the tiny val split by redistributing from train.
    """
    raw_dir = Path("data/raw/chest_xray")
    if not raw_dir.exists():
        print(f"Expected folder {raw_dir} not found. Check download.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        src = raw_dir / split
        dst = DATA_DIR / split
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"Copied {split} → {dst}")

    print("Reorganisation complete.")
    _fix_val_split()


def _fix_val_split():
    """
    Move 20% of training images into val to create a proper validation set.
    The original val set has only 8 images per class — too small for meaningful metrics.
    """
    print("Redistributing val split (20% of train)...")
    random.seed(42)

    for class_name in ["NORMAL", "PNEUMONIA"]:
        train_dir = DATA_DIR / "train" / class_name
        val_dir = DATA_DIR / "val" / class_name
        val_dir.mkdir(parents=True, exist_ok=True)

        # Remove existing tiny val images
        for f in val_dir.glob("*.jpeg"):
            f.unlink()

        all_images = list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.jpg"))
        random.shuffle(all_images)
        val_count = int(len(all_images) * 0.20)
        val_images = all_images[:val_count]

        for img in val_images:
            shutil.move(str(img), str(val_dir / img.name))

        print(f"  {class_name}: {val_count} images moved to val")

    print("Val split fixed.")


def print_stats():
    print("\nDataset summary:")
    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            count = len(list(class_dir.glob("*")))
            print(f"  {split}/{class_dir.name}: {count} images")


if __name__ == "__main__":
    download()
    reorganise()
    print_stats()
    print("\nDone. Run training with: python -m training.pneumonia.train")
