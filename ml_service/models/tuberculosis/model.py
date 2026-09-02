"""
models/tuberculosis/model.py
Tuberculosis classifier — reuses the exact DenseNet-121 architecture
established for Pneumonia. Same backbone, same preprocessing, same
Grad-CAM target layer. Only the classification head and dataset differ.

Task:
  Binary classification on chest X-ray images.
  Classes:
    0 → NORMAL
    1 → TUBERCULOSIS

Dataset:
  TBX11K (11,200 chest X-rays with TB annotations)
  https://www.kaggle.com/datasets/usmanshams/tbx-11
  Montgomery + Shenzhen sets:
  https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from core.logger import logger
from core.config import settings


CLASS_NAMES  = ["NORMAL", "TUBERCULOSIS"]
NUM_CLASSES  = len(CLASS_NAMES)
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class TuberculosisClassifier(nn.Module):
    """DenseNet-121 backbone — identical structure to PneumoniaClassifier."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = backbone.features

        in_features = backbone.classifier.in_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = torch.relu(features)
        return self.classifier(features)


def build_model() -> TuberculosisClassifier:
    model = TuberculosisClassifier()
    logger.info(f"Built TuberculosisClassifier | classes={CLASS_NAMES}")
    return model


def load_trained_model(
    model_path: Path | None = None,
    device: str | None = None,
) -> TuberculosisClassifier:
    device = device or settings.device
    model_path = model_path or (
        settings.model_registry_dir / "tuberculosis" / "best_model.pth"
    )

    model = TuberculosisClassifier()

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded TB model from {model_path} on {device}")
    else:
        logger.warning(
            f"No trained TB model found at {model_path}. "
            "Using ImageNet backbone weights only — run training first."
        )

    model = model.to(device)
    model.eval()
    return model
