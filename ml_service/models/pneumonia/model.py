# Pneumonia Classifier
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from core.logger import logger
from core.config import settings


CLASS_NAMES = ["NORMAL", "PNEUMONIA", "COVID-19"]
NUM_CLASSES = len(CLASS_NAMES)
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class PneumoniaClassifier(nn.Module):
    """
    DenseNet-121 backbone with a custom classification head.
    Pretrained weights from ImageNet are loaded by default.
    The original classifier is replaced with a 3-class head.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        # Load DenseNet-121 pretrained on ImageNet
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # Keep all feature layers, remove the original classifier
        self.features = backbone.features

        # Custom head: GlobalAvgPool → Dropout → Linear → Softmax
        in_features = backbone.classifier.in_features  # 1024 for DenseNet-121
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
        # DenseNet features end with a ReLU via bn → activate before pooling
        features = torch.relu(features)
        out = self.classifier(features)
        return out  # raw logits — apply softmax for probabilities


def build_model(pretrained_backbone: bool = True) -> PneumoniaClassifier:
    """Build a fresh PneumoniaClassifier. Use for training."""
    model = PneumoniaClassifier()
    logger.info(
        f"Built PneumoniaClassifier | classes={CLASS_NAMES} | "
        f"pretrained_backbone={pretrained_backbone}"
    )
    return model


def load_trained_model(
    model_path: Path | None = None,
    device: str | None = None,
) -> PneumoniaClassifier:
    """
    Load a fine-tuned model from a .pth checkpoint.
    Falls back to backbone-only weights if checkpoint not found (useful in dev).
    """
    device = device or settings.device
    model_path = model_path or settings.pneumonia_model_path

    model = PneumoniaClassifier()

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Support both raw state_dict and wrapped checkpoint dicts
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded pneumonia model from {model_path} on {device}")
    else:
        logger.warning(
            f"No trained model found at {model_path}. "
            "Using ImageNet backbone weights only — run training first."
        )

    model = model.to(device)
    model.eval()
    return model
