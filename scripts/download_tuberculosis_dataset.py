"""
scripts/download_tuberculosis_dataset.py
Downloads the Shenzhen + Montgomery TB chest X-ray datasets from Kaggle
and creates a train/val/test split.

Dataset:
  Kaggle: https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen
  ~660 images (Shenzhen set) — small but well-curated and widely benchmarked.

  For a larger dataset, also consider TBX11K:
  https://www.kaggle.com/datasets/usmanshams/tbx-11

This script handles the Shenzhen set, which ships as:
  images/  (all X-rays, filename pattern: CHNCXN_<id>_<label>.png)
  ClinicalReadings/  (text reports)

Label convention: filename ending in "_0.png" = NORMAL, "_1.png" = TUBERCULOSIS

Output structure:
  data\\raw\\tuberculosis\\
    train\\NORMAL\\  , train\\TUBERCULOSIS\\
    val\\NORMAL\\    , val\\TUBERCULOSIS\\
    test\\NORMAL\\   , test\\TUBERCULOSIS\\
"""
import os
import shutil
import random
from pathlib import Path

DATA_DIR       = Path("data/raw/tuberculosis")
KAGGLE_DATASET = "raddar/tuberculosis-chest-xrays-shenzhen"
SPLIT_RATIOS   = {"train": 0.70, "val": 0.15, "test": 0.15}


def download():
    print("Downloading TB Chest X-Ray (Shenzhen) dataset from Kaggle...")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p data/raw/ --unzip")
    print("Download complete.")


def split_and_organise():
    """
    Locate extracted images, label by filename suffix, and split into
    train/val/test class folders.
    """
    candidates = [
        Path("data/raw/images/images"),
        Path("data/raw/images"),
        Path("data/raw/ChinaSet_AllFiles/CXR_png"),
        Path("data/raw/Shenzhen"),
    ]
    raw_dir = next((c for c in candidates if c.exists()), None)
    if raw_dir is None:
        print("Could not find extracted image folder. Check data/raw/ contents manually.")
        return

    random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_images = list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpg"))
    normal_images, tb_images = [], []

    for img in all_images:
        stem = img.stem
        if stem.endswith("_0"):
            normal_images.append(img)
        elif stem.endswith("_1"):
            tb_images.append(img)

    print(f"Found {len(normal_images)} NORMAL, {len(tb_images)} TUBERCULOSIS images")

    for class_name, images in [("NORMAL", normal_images), ("TUBERCULOSIS", tb_images)]:
        random.shuffle(images)
        total = len(images)
        n_train = int(total * SPLIT_RATIOS["train"])
        n_val   = int(total * SPLIT_RATIOS["val"])

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dest_dir = DATA_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy2(img, dest_dir / img.name)

        print(f"{class_name}: {n_train} train | {n_val} val | {total - n_train - n_val} test")

    print("Organisation complete.")


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
    #download()
    split_and_organise()
    print_stats()
    print("\nDone. Run training with:")
    print("  python -m training.tuberculosis.train")
