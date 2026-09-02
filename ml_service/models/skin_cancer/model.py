"""
models/skin_cancer/model.py
Skin lesion classifier using EfficientNet-B3.

Task:
  Multi-class classification of dermoscopic skin lesion images.
  7 classes from the HAM10000 dataset:
    0 → MELANOMA (mel)             — malignant, most dangerous
    1 → MELANOCYTIC_NEVUS (nv)     — benign mole
    2 → BASAL_CELL_CARCINOMA (bcc) — malignant
    3 → ACTINIC_KERATOSIS (akiec)  — pre-cancerous
    4 → BENIGN_KERATOSIS (bkl)     — benign
    5 → DERMATOFIBROMA (df)        — benign
    6 → VASCULAR_LESION (vasc)     — benign

Why EfficientNet-B3 (not B0 like Malaria):
  - Skin lesion images carry finer-grained diagnostic detail than blood
    smear cells (asymmetry, border irregularity, colour variation, texture)
  - B3 has a larger input resolution (300x300) and more capacity —
    standard choice in published HAM10000 benchmarks
  - Still much lighter than ResNet-50, so inference stays fast

Dataset:
  HAM10000 ("Human Against Machine with 10000 training images")
  https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
  10,015 images | 7 classes | Significant class imbalance
  (nv: 6,705 vs df: 115 — ~58x imbalance, requires weighted sampling)
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from core.logger import logger
from core.config import settings


CLASS_NAMES = [
    "MELANOMA",
    "MELANOCYTIC_NEVUS",
    "BASAL_CELL_CARCINOMA",
    "ACTINIC_KERATOSIS",
    "BENIGN_KERATOSIS",
    "DERMATOFIBROMA",
    "VASCULAR_LESION",
]
NUM_CLASSES  = len(CLASS_NAMES)
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Malignant/pre-malignant classes — these always escalate triage
MALIGNANT_CLASSES = {"MELANOMA", "BASAL_CELL_CARCINOMA", "ACTINIC_KERATOSIS"}

SKIN_CANCER_IMAGE_SIZE = 300   # EfficientNet-B3's native resolution


class SkinCancerClassifier(nn.Module):
    """
    EfficientNet-B3 backbone with a custom 7-class classification head.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()

        backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        in_features = backbone.classifier[1].in_features  # 1536 for B3
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model() -> SkinCancerClassifier:
    model = SkinCancerClassifier()
    logger.info(f"Built SkinCancerClassifier | classes={CLASS_NAMES}")
    return model


def load_trained_model(
    model_path: Path | None = None,
    device: str | None = None,
) -> SkinCancerClassifier:
    device = device or settings.device
    model_path = model_path or (
        settings.model_registry_dir / "skin_cancer" / "best_model.pth"
    )

    model = SkinCancerClassifier()

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded skin cancer model from {model_path} on {device}")
    else:
        logger.warning(
            f"No trained skin cancer model found at {model_path}. "
            "Using ImageNet backbone weights only — run training first."
        )

    model = model.to(device)
    model.eval()
    return model
