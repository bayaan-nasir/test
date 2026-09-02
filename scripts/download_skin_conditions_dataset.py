import os
import shutil
import random
from pathlib import Path

DATA_DIR = Path("data/raw/skin_conditions")
KAGGLE_DATASET = "shubhamgoel27/dermnet"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Exact folder names from the DermNet dataset
CLASS_FOLDER_MAP = {
    "Eczema Photos": "ECZEMA",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "RINGWORM",
    "Psoriasis pictures Lichen Planus and related diseases": "PSORIASIS",
    "Acne and Rosacea Photos": "ACNE",
}


def download():
    print("Downloading skin conditions dataset from Kaggle (~2GB)...")
    # Using -q to keep logs clean, --unzip to extract immediately
    os.system(
        f"kaggle datasets download -d {KAGGLE_DATASET} -p data/raw/ --unzip")
    print("Download complete.")


def split_and_organise():
    random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for source_name, target_class in CLASS_FOLDER_MAP.items():
        all_images = []

        # Aggressively hunt for the source folders anywhere in data/raw
        # This solves the issue of DermNet having nested train/test folders
        for matched_dir in Path("data/raw").rglob(source_name):
            # Prevent recursive reading if we run the script twice
            if "skin_conditions" not in matched_dir.parts:
                all_images.extend(list(matched_dir.rglob("*.jpg")))
                all_images.extend(list(matched_dir.rglob("*.jpeg")))
                all_images.extend(list(matched_dir.rglob("*.png")))

        if not all_images:
            print(
                f"Warning: No images found for '{source_name}'. Kaggle download may have failed.")
            continue

        # Remove duplicates just in case
        all_images = list(set(all_images))
        random.shuffle(all_images)

        total = len(all_images)
        n_train = int(total * SPLIT_RATIOS["train"])
        n_val = int(total * SPLIT_RATIOS["val"])

        splits = {
            "train": all_images[:n_train],
            "val":   all_images[n_train:n_train + n_val],
            "test":  all_images[n_train + n_val:],
        }

        for split_name, images in splits.items():
            dest_dir = DATA_DIR / split_name / target_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                shutil.copy2(img, dest_dir / img.name)

        print(
            f"✓ {target_class}: {n_train} train | {n_val} val | {total - n_train - n_val} test (of {total})")


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
    print("  python -m ml_service.training.skin_conditions.train")
