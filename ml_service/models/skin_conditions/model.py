"""
models/skin_conditions/model.py
Skin Conditions classifier (Eczema, Ringworm, and related conditions)
using EfficientNet-B3 — reuses the exact architecture established for
Skin Cancer, but trained on a separate dataset and class set.

Why this is a SEPARATE model from Skin Cancer, not an extra class on it:
  Skin Cancer (HAM10000) and general dermatology conditions (Eczema/
  Ringworm/Psoriasis datasets) come from entirely different dataset
  sources with different image conventions (dermoscopic close-ups vs.
  regular smartphone photos of affected skin areas). Combining them
  into one classifier would mix incompatible image distributions.
  Keeping them as two separate models keeps each one's training data
  consistent and makes per-disease evaluation cleaner for the thesis.

Task:
  Multi-class classification of common dermatological skin conditions.
  Classes (5, common-skin-disease Kaggle datasets typically include):
    0 → ECZEMA
    1 → RINGWORM (Tinea)
    2 → PSORIASIS
    3 → ACNE
    4 → NORMAL_SKIN

Dataset:
  Kaggle "Skin Diseases Image Dataset" or DermNet-derived subsets
  https://www.kaggle.com/datasets/shubhamgoel27/dermnet
  Several thousand images per class, smartphone-photo quality
  (lower resolution and more variable lighting than HAM10000's
  dermoscopic images — reflected in the augmentation strategy below)
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from core.logger import logger
from core.config import settings


CLASS_NAMES = [
    "ECZEMA",
    "RINGWORM",
    "PSORIASIS",
    "ACNE",
    "NORMAL_SKIN",
]
NUM_CLASSES  = len(CLASS_NAMES)
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

SKIN_CONDITION_IMAGE_SIZE = 300   # same as Skin Cancer — EfficientNet-B3 native res


class SkinConditionsClassifier(nn.Module):
    """
    EfficientNet-B3 backbone with a 5-class classification head.
    Architecturally identical to SkinCancerClassifier — only the
    final layer size and training data differ.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()

        backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        in_features = backbone.classifier[1].in_features
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


def build_model() -> SkinConditionsClassifier:
    model = SkinConditionsClassifier()
    logger.info(f"Built SkinConditionsClassifier | classes={CLASS_NAMES}")
    return model


def load_trained_model(
    model_path: Path | None = None,
    device: str | None = None,
) -> SkinConditionsClassifier:
    device = device or settings.device
    model_path = model_path or (
        settings.model_registry_dir / "skin_conditions" / "best_model.pth"
    )

    model = SkinConditionsClassifier()

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded skin conditions model from {model_path} on {device}")
    else:
        logger.warning(
            f"No trained skin conditions model found at {model_path}. "
            "Using ImageNet backbone weights only — run training first."
        )

    model = model.to(device)
    model.eval()
    return model
