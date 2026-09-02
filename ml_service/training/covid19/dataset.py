# COVID-19 Dataset Loader
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from models.covid19.model import CLASS_TO_IDX
from preprocessing.image_transforms import get_train_transforms, get_val_transforms
from core.logger import logger


class Covid19Dataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str = "train"):
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = (
            get_train_transforms() if split == "train" else get_val_transforms()
        )
        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        valid_exts = {".jpg", ".jpeg", ".png"}
        for class_name, label_idx in CLASS_TO_IDX.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class folder not found: {class_dir} — skipping")
                continue
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in valid_exts]
            for img_path in images:
                self.samples.append((img_path, label_idx))
        logger.info(f"Covid19Dataset [{self.split}] — {len(self.samples)} images loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)
        transformed = self.transform(image=img_array)
        return transformed["image"], label

    def get_class_weights(self) -> torch.Tensor:
        labels = [label for _, label in self.samples]
        counts = np.bincount(labels, minlength=len(CLASS_TO_IDX))
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    datasets = {
        split: Covid19Dataset(data_dir, split)
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
