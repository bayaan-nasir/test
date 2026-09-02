"""
scripts/download_breast_cancer_dataset.py
Downloads the BreakHis breast histopathology dataset from Kaggle and
reorganises the nested magnification-level folders into a flat
BENIGN / MALIGNANT train/val/test structure.

Dataset:
  Kaggle: https://www.kaggle.com/datasets/ambarish/breakhis
  7,909 images | All magnifications (40X, 100X, 200X, 400X) combined

Original BreakHis structure (deeply nested):
  BreaKHis_v1/histology_slides/breast/
    benign/SOB/<subtype>/<patient_id>/<magnification>/*.png
    malignant/SOB/<subtype>/<patient_id>/<magnification>/*.png

This script walks the entire tree, collects every image regardless of
subtype/magnification, and creates a flat split:
  data\\raw\\breast_cancer\\
    train\\BENIGN\\   , train\\MALIGNANT\\
    val\\BENIGN\\     , val\\MALIGNANT\\
    test\\BENIGN\\    , test\\MALIGNANT\\
"""
import os
import shutil
import random
from pathlib import Path

DATA_DIR       = Path("data/raw/breast_cancer")
KAGGLE_DATASET = "ambarish/breakhis"
SPLIT_RATIOS   = {"train": 0.70, "val": 0.15, "test": 0.15}


def download():
    print("Downloading BreakHis dataset from Kaggle (this is ~3-4GB, may take a while)...")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p data/raw/ --unzip")
    print("Download complete.")


def find_breakhis_root() -> Path | None:
    """Locate the histology_slides/breast root regardless of exact extraction path."""
    for candidate in Path("data/raw").rglob("breast"):
        if (candidate / "benign").exists() and (candidate / "malignant").exists():
            return candidate
    return None


def split_and_organise():
    root = find_breakhis_root()
    if root is None:
        print("Could not locate BreaKHis 'breast' folder. Check data/raw/ contents manually.")
        return

    random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for source_label, class_name in [("benign", "BENIGN"), ("malignant", "MALIGNANT")]:
        class_root = root / source_label
        # Recursively collect every .png regardless of subtype/patient/magnification
        all_images = list(class_root.rglob("*.png"))
        random.shuffle(all_images)

        total = len(all_images)
        n_train = int(total * SPLIT_RATIOS["train"])
        n_val   = int(total * SPLIT_RATIOS["val"])

        splits = {
            "train": all_images[:n_train],
            "val":   all_images[n_train:n_train + n_val],
            "test":  all_images[n_train + n_val:],
        }

        for split_name, images in splits.items():
            dest_dir = DATA_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                # Prefix with patient folder name to avoid filename collisions
                unique_name = f"{img.parent.parent.name}_{img.name}"
                shutil.copy2(img, dest_dir / unique_name)

        print(f"{class_name}: {n_train} train | {n_val} val | {total - n_train - n_val} test (of {total} total)")

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
    download()
    split_and_organise()
    print_stats()
    print("\nDone. Run training with:")
    print("  python -m training.breast_cancer.train")
