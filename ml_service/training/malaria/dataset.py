# Dataset class for malaria
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.malaria.model import CLASS_TO_IDX
from preprocessing.malaria_transforms import (
    get_malaria_train_transforms,
    get_malaria_val_transforms,
)
from core.logger import logger


class MalariaDataset(Dataset):
    """
    Loads blood smear cell images from class subfolders.
    Folder names are matched case-insensitively to CLASS_TO_IDX keys.
    """

    # Map folder name variants → canonical class key
    FOLDER_MAP = {
        "parasitized": "PARASITIZED",
        "uninfected":  "UNINFECTED",
    }

    def __init__(self, data_dir: str | Path, split: str = "train"):
        self.data_dir = Path(data_dir) / split
        self.split    = split
        self.transform = (
            get_malaria_train_transforms()
            if split == "train"
            else get_malaria_val_transforms()
        )
        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        valid_exts = {".png", ".jpg", ".jpeg"}

        for folder in self.data_dir.iterdir():
            if not folder.is_dir():
                continue

            canonical = self.FOLDER_MAP.get(folder.name.lower())
            if canonical is None:
                logger.warning(f"Unknown folder '{folder.name}' — skipping")
                continue

            label_idx = CLASS_TO_IDX[canonical]
            images = [
                p for p in folder.iterdir()
                if p.suffix.lower() in valid_exts
            ]
            for img_path in images:
                self.samples.append((img_path, label_idx))

        logger.info(
            f"MalariaDataset [{self.split}] — "
            f"{len(self.samples)} images loaded"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)
        transformed = self.transform(image=img_array)
        return transformed["image"], label


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """
    Build train, val, test DataLoaders for malaria.
    No weighted sampler needed — dataset is perfectly balanced.

    Larger batch size than pneumonia (64 vs 32) because:
      - Cell images are much smaller (128x128 vs 224x224)
      - Balanced dataset — no need for weighted sampling
    """
    datasets = {
        split: MalariaDataset(data_dir, split)
        for split in ["train", "val", "test"]
    }

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        for split in ["train", "val", "test"]
    }

    return loaders
