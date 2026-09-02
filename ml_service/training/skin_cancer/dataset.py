"""
training/skin_cancer/dataset.py
PyTorch Dataset for the HAM10000 skin lesion dataset.

Unlike Pneumonia/Malaria/TB (folder-per-class), HAM10000 ships as:
  - Two image folders: HAM10000_images_part_1/, HAM10000_images_part_2/
  - One metadata CSV: HAM10000_metadata.csv with columns:
      image_id, dx (diagnosis code), dx_type, age, sex, localization

This Dataset reads the CSV to map image_id -> diagnosis label,
then locates the actual file across both image folders.

Expected structure after download:
  data/raw/skin_cancer/
    HAM10000_images_part_1/*.jpg
    HAM10000_images_part_2/*.jpg
    HAM10000_metadata.csv
    train_split.csv   (created by download script — train/val/test assignment)
    val_split.csv
    test_split.csv

dx code -> our class mapping:
  mel   -> MELANOMA
  nv    -> MELANOCYTIC_NEVUS
  bcc   -> BASAL_CELL_CARCINOMA
  akiec -> ACTINIC_KERATOSIS
  bkl   -> BENIGN_KERATOSIS
  df    -> DERMATOFIBROMA
  vasc  -> VASCULAR_LESION
"""
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from models.skin_cancer.model import CLASS_TO_IDX
from preprocessing.skin_cancer_transforms import get_skin_train_transforms, get_skin_val_transforms
from core.logger import logger


DX_CODE_TO_CLASS = {
    "mel":   "MELANOMA",
    "nv":    "MELANOCYTIC_NEVUS",
    "bcc":   "BASAL_CELL_CARCINOMA",
    "akiec": "ACTINIC_KERATOSIS",
    "bkl":   "BENIGN_KERATOSIS",
    "df":    "DERMATOFIBROMA",
    "vasc":  "VASCULAR_LESION",
}


class SkinCancerDataset(Dataset):
    """
    Loads HAM10000 images using a split CSV (image_id, dx) and locates
    the actual .jpg file across both image part folders.
    """

    def __init__(self, data_dir: str | Path, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = (
            get_skin_train_transforms() if split == "train" else get_skin_val_transforms()
        )

        # Build a lookup of image_id -> full path across both image folders
        self.image_dirs = [
            self.data_dir / "HAM10000_images_part_1",
            self.data_dir / "HAM10000_images_part_2",
        ]
        self._image_path_cache: dict[str, Path] = {}
        for img_dir in self.image_dirs:
            if img_dir.exists():
                for p in img_dir.glob("*.jpg"):
                    self._image_path_cache[p.stem] = p

        split_csv = self.data_dir / f"{split}_split.csv"
        if not split_csv.exists():
            raise FileNotFoundError(
                f"{split_csv} not found. Run scripts\\download_skin_cancer_dataset.py first."
            )

        df = pd.read_csv(split_csv)
        self.samples: list[tuple[Path, int]] = []

        for _, row in df.iterrows():
            image_id = row["image_id"]
            dx_code  = row["dx"]
            class_name = DX_CODE_TO_CLASS.get(dx_code)
            if class_name is None:
                continue
            img_path = self._image_path_cache.get(image_id)
            if img_path is None:
                logger.warning(f"Image not found for id={image_id} — skipping")
                continue
            self.samples.append((img_path, CLASS_TO_IDX[class_name]))

        logger.info(f"SkinCancerDataset [{split}] — {len(self.samples)} images loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)
        transformed = self.transform(image=img_array)
        return transformed["image"], label

    def get_class_weights(self) -> torch.Tensor:
        """
        HAM10000 has severe class imbalance (nv: ~67%, df: ~1%).
        Inverse-frequency weighting is essential here.
        """
        labels = [label for _, label in self.samples]
        counts = np.bincount(labels, minlength=len(CLASS_TO_IDX))
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 16,   # smaller batch — 300x300 images use more memory
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    datasets = {
        split: SkinCancerDataset(data_dir, split)
        for split in ["train", "val", "test"]
    }

    class_weights = datasets["train"].get_class_weights()
    sample_weights = [
        class_weights[label].item() for _, label in datasets["train"].samples
    ]
    train_sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }
    return loaders
