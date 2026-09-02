"""
models/breast_cancer/model.py
Breast Cancer classifier using ResNet-50 fine-tuned on histopathology images.

Task:
  Binary classification of breast tissue histology slide patches.
  Classes:
    0 → BENIGN
    1 → MALIGNANT

Why ResNet-50 (not EfficientNet):
  - Histology images have very different texture statistics than natural
    photos or X-rays — dense cellular structures, staining artefacts
  - ResNet-50 is the most widely benchmarked architecture on BreakHis
    in published literature — easier to compare/justify results in
    a thesis defense
  - Residual connections handle the deeper, more complex patterns in
    histopathology slides better than shallower nets at this dataset size

Dataset:
  BreakHis (Breast Cancer Histopathological Database)
  https://www.kaggle.com/datasets/ambarish/breakhis
  7,909 images | Patient-level binary labels (benign/malignant)
  Images come at 4 magnification levels (40X, 100X, 200X, 400X) —
  we train on all magnifications combined for better generalisation.
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from core.logger import logger
from core.config import settings


CLASS_NAMES  = ["BENIGN", "MALIGNANT"]
NUM_CLASSES  = len(CLASS_NAMES)
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

BREAST_IMAGE_SIZE = 224   # ResNet-50's native resolution


class BreastCancerClassifier(nn.Module):
    """
    ResNet-50 backbone with a custom binary classification head.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Keep everything except the final fully-connected layer
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # up to last conv block
        self.avgpool  = backbone.avgpool

        in_features = backbone.fc.in_features  # 2048 for ResNet-50
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
        return self.classifier(x)


def build_model() -> BreastCancerClassifier:
    model = BreastCancerClassifier()
    logger.info(f"Built BreastCancerClassifier | classes={CLASS_NAMES}")
    return model


def load_trained_model(
    model_path: Path | None = None,
    device: str | None = None,
) -> BreastCancerClassifier:
    device = device or settings.device
    model_path = model_path or (
        settings.model_registry_dir / "breast_cancer" / "best_model.pth"
    )

    model = BreastCancerClassifier()

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded breast cancer model from {model_path} on {device}")
    else:
        logger.warning(
            f"No trained breast cancer model found at {model_path}. "
            "Using ImageNet backbone weights only — run training first."
        )

    model = model.to(device)
    model.eval()
    return model
