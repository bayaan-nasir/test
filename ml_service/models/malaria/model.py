# Malaria Cell Classifier
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from core.logger import logger
from core.config import settings


CLASS_NAMES  = ["UNINFECTED", "PARASITIZED"]
NUM_CLASSES  = len(CLASS_NAMES)
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class MalariaClassifier(nn.Module):
    """
    EfficientNet-B0 backbone with a custom binary classification head.

    Architecture:
      EfficientNet-B0 features → GlobalAvgPool (built-in) →
      Dropout(0.4) → Linear(1280, 256) → ReLU →
      Dropout(0.2) → Linear(256, 2)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Keep all feature extraction layers
        self.features = backbone.features
        self.avgpool  = backbone.avgpool   # AdaptiveAvgPool2d(1, 1)

        # Replace classifier
        in_features = backbone.classifier[1].in_features  # 1280 for B0
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x   # raw logits


def build_model() -> MalariaClassifier:
    """Build a fresh MalariaClassifier with ImageNet backbone weights."""
    model = MalariaClassifier()
    logger.info(f"Built MalariaClassifier | classes={CLASS_NAMES}")
    return model


def load_trained_model(
    model_path: Path | None = None,
    device: str | None = None,
) -> MalariaClassifier:
    """Load a fine-tuned malaria model from checkpoint."""
    device = device or settings.device
    model_path = model_path or (
        settings.model_registry_dir / "malaria" / "best_model.pth"
    )

    model = MalariaClassifier()

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded malaria model from {model_path} on {device}")
    else:
        logger.warning(
            f"No trained malaria model found at {model_path}. "
            "Using ImageNet backbone weights only — run training first."
        )

    model = model.to(device)
    model.eval()
    return model
