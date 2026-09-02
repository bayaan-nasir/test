"""
Downloads the COVID-19 Radiography Database from Kaggle and creates a
3-class (NORMAL / COVID-19 / PNEUMONIA) train/val/test split.

Dataset:
  Kaggle: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
  21,000 images across 4 original folders: COVID, Normal, Viral Pneumonia,
  Lung_Opacity. For this 3-class setup, Viral Pneumonia and Lung_Opacity
  are merged into a single PNEUMONIA class.

Output structure:
  data\\raw\\covid19\\
    train\\NORMAL\\    , train\\COVID-19\\    , train\\PNEUMONIA\\
    val\\NORMAL\\      , val\\COVID-19\\      , val\\PNEUMONIA\\
    test\\NORMAL\\     , test\\COVID-19\\     , test\\PNEUMONIA\\
"""
import os
import shutil
import random
from pathlib import Path

DATA_DIR       = Path("data/raw/covid19")
KAGGLE_DATASET = "tawsifurrahman/covid19-radiography-database"
SPLIT_RATIOS   = {"train": 0.70, "val": 0.15, "test": 0.15}

# Maps original dataset folder names -> our 3-class scheme
SOURCE_FOLDER_MAP = {
    "Normal":          "NORMAL",
    "COVID":           "COVID-19",
    "Viral Pneumonia": "PNEUMONIA",
    "Lung_Opacity":    "PNEUMONIA",   # merged into PNEUMONIA
}


def download():
    print("Downloading COVID-19 Radiography Database from Kaggle (~1-2GB)...")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p data/raw/ --unzip")
    print("Download complete.")


def find_dataset_root() -> Path | None:
    """The dataset extracts with a nested structure — locate the actual class folders."""
    for candidate in Path("data/raw").rglob("*"):
        if candidate.is_dir() and candidate.name in SOURCE_FOLDER_MAP:
            return candidate.parent
    return None


def split_and_organise():
    root = find_dataset_root()
    if root is None:
        print("Could not locate the COVID-19 dataset class folders. Check data/raw/ manually.")
        return

    random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Collect images per target class, merging source folders as needed
    images_by_target_class: dict[str, list[Path]] = {"NORMAL": [], "COVID-19": [], "PNEUMONIA": []}

    for source_name, target_class in SOURCE_FOLDER_MAP.items():
        source_dir = root / source_name / "images"
        if not source_dir.exists():
            source_dir = root / source_name   # some releases skip the "images" subfolder
        if not source_dir.exists():
            print(f"Warning: {source_dir} not found — skipping")
            continue
        imgs = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg"))
        images_by_target_class[target_class].extend(imgs)

    for target_class, images in images_by_target_class.items():
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
            dest_dir = DATA_DIR / split_name / target_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                # Prefix with source folder name to avoid collisions after merging
                unique_name = f"{img.parent.parent.name}_{img.name}"
                shutil.copy2(img, dest_dir / unique_name)

        print(f"{target_class}: {n_train} train | {n_val} val | {total - n_train - n_val} test (of {total})")

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
    print("  python -m training.covid19.train")
