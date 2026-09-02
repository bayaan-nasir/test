"""
training/tuberculosis/dataset.py
PyTorch Dataset for TB chest X-ray classification.
Structurally identical to PneumoniaDataset.

Expected folder structure:
  data/raw/tuberculosis/
    train/NORMAL/ , train/TUBERCULOSIS/
    val/NORMAL/   , val/TUBERCULOSIS/
    test/NORMAL/  , test/TUBERCULOSIS/
"""
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from models.tuberculosis.model import CLASS_TO_IDX
from preprocessing.image_transforms import get_train_transforms, get_val_transforms
from core.logger import logger


class TuberculosisDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str = "train"):
        self.data_dir = Path(data_dir).resolve() / split
        self.split = split
        self.transform = (
            get_train_transforms() if split == "train" else get_val_transforms()
        )
        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        valid_exts = {".jpg", ".jpeg", ".png"}

        # THESE TWO LINES FOR DEBUGGING:
        print(f"\n--- DEBUG PATH CHECK ---")
        print(f"Dataset is searching in: {self.data_dir.resolve()}\n")
        
        # Ensure base directory exists
        if not self.data_dir.exists():
            logger.error(f"Split directory not found: {self.data_dir}")
            return

        # Map actual directories on disk case-insensitively to solve exact string matching issues
        disk_dirs = {p.name.upper(): p for p in self.data_dir.iterdir() if p.is_dir()}

        for class_name, label_idx in CLASS_TO_IDX.items():
            # Look up target folder matching upper case name structure
            class_dir = disk_dirs.get(class_name.upper())

            if class_dir is None or not class_dir.exists():
                logger.warning(f"Class folder not found for {class_name} in {self.data_dir} — skipping")
                continue
                
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in valid_exts]
            for img_path in images:
                self.samples.append((img_path, label_idx))
                
        logger.info(f"TuberculosisDataset [{self.split}] — {len(self.samples)} images loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)
        transformed = self.transform(image=img_array)
        return transformed["image"], label

    def get_class_weights(self) -> torch.Tensor:
        if len(self.samples) == 0:
            return torch.tensor([1.0, 1.0], dtype=torch.float32)
        labels = [label for _, label in self.samples]
        counts = np.bincount(labels, minlength=len(CLASS_TO_IDX))
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> dict[str, DataLoader]:
    datasets = {
        split: TuberculosisDataset(data_dir, split)
        for split in ["train", "val", "test"]
    }

    train_sampler = None
    if use_weighted_sampler and len(datasets["train"].samples) > 0:
        class_weights = datasets["train"].get_class_weights()
        sample_weights = [
            class_weights[label].item() for _, label in datasets["train"].samples
        ]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

    # Use 0 workers if no samples found to prevent worker hanging during fallback testing
    actual_workers = num_workers if len(datasets["train"].samples) > 0 else 0

    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, sampler=train_sampler,
            shuffle=(train_sampler is None), num_workers=actual_workers, pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False,
            num_workers=actual_workers, pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False,
            num_workers=actual_workers, pin_memory=True,
        ),
    }
    return loaders