"""
scripts/download_normal_skin.py
Downloads a supplementary dataset containing healthy skin images
to serve as the NORMAL_SKIN class for our classifier.
"""
import os
import shutil
import random
from pathlib import Path

# A viral skin lesion dataset that includes a high-quality "Healthy" skin class
KAGGLE_DATASET = "borhan2003/multi-class-viral-skin-lesion-dataset-mcvsld"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def get_normal_skin():
    print("Downloading Healthy Skin dataset from Kaggle...")
    os.system(
        f"kaggle datasets download -d {KAGGLE_DATASET} -p data/raw/temp_healthy --unzip")

    # Find the healthy folder(s)
    normal_images = []
    for candidate in Path("data/raw/temp_healthy").rglob("*"):
        if candidate.is_dir() and candidate.name.lower() == "healthy":
            normal_images.extend(list(candidate.rglob("*.jpg")))
            normal_images.extend(list(candidate.rglob("*.jpeg")))
            normal_images.extend(list(candidate.rglob("*.png")))

    if not normal_images:
        print("Could not find 'Healthy' skin images in the downloaded dataset.")
        return

    # Deduplicate and shuffle
    normal_images = list(set(normal_images))
    random.seed(42)
    random.shuffle(normal_images)

    total = len(normal_images)
    n_train = int(total * SPLIT_RATIOS["train"])
    n_val = int(total * SPLIT_RATIOS["val"])

    splits = {
        "train": normal_images[:n_train],
        "val":   normal_images[n_train:n_train + n_val],
        "test":  normal_images[n_train + n_val:],
    }

    print("\nExtracting and organizing NORMAL_SKIN images...")
    for split_name, images in splits.items():
        dest_dir = Path(f"data/raw/skin_conditions/{split_name}/NORMAL_SKIN")
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img in images:
            shutil.copy2(img, dest_dir / img.name)

    print(
        f"✓ NORMAL_SKIN: {n_train} train | {n_val} val | {len(splits['test'])} test")

    # Cleanup temporary download
    print("Cleaning up temporary files...")
    shutil.rmtree("data/raw/temp_healthy", ignore_errors=True)
    print("Done! Your NORMAL_SKIN class is now populated.")


if __name__ == "__main__":
    get_normal_skin()
