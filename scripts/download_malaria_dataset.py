# Script to download and prepare the Malaria Cell Images dataset from Kaggle.
# The dataset is split into train/val/test folders with a 70/15/15
import os
import shutil
import random
from pathlib import Path

DATA_DIR       = Path("data/raw/malaria")
KAGGLE_DATASET = "iarunava/cell-images-for-detecting-malaria"
SPLIT_RATIOS   = {"train": 0.70, "val": 0.15, "test": 0.15}
CLASS_FOLDERS  = {"Parasitized": "PARASITIZED", "Uninfected": "UNINFECTED"}


def download():
    print("Downloading Malaria Cell Images dataset from Kaggle...")
    os.system(
        f"kaggle datasets download -d {KAGGLE_DATASET} "
        f"-p data/raw/ --unzip"
    )
    print("Download complete.")


def split_and_organise():
    """
    The Kaggle dataset extracts to data/raw/cell_images/ with
    two subfolders: Parasitized/ and Uninfected/.
    This splits each into train/val/test.
    """
    raw_dir = Path("data/raw/cell_images")
    if not raw_dir.exists():
        # Try alternate extraction paths
        for candidate in [Path("data/raw/Cell_Images"), Path("data/raw/cell-images")]:
            if candidate.exists():
                raw_dir = candidate
                break
        else:
            print(f"Could not find extracted dataset folder. Check data/raw/")
            return

    random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for folder_name in ["Parasitized", "Uninfected"]:
        src_dir = raw_dir / folder_name
        if not src_dir.exists():
            print(f"Warning: {src_dir} not found — skipping")
            continue

        all_images = [
            p for p in src_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
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
            dest_dir = DATA_DIR / split_name / folder_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                shutil.copy2(img, dest_dir / img.name)

        print(
            f"{folder_name}: {n_train} train | "
            f"{n_val} val | {total - n_train - n_val} test"
        )

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
    print("\nDone. Run training with: python -m training.malaria.train")
