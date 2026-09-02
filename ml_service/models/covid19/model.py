"""
models/covid19/model.py
COVID-19 classifier — reuses the DenseNet-121 chest X-ray pipeline
established for Pneumonia and Tuberculosis.

Task:
  3-class classification on chest X-ray images.
  Classes:
    0 → NORMAL
    1 → COVID-19
    2 → PNEUMONIA (non-COVID viral/bacterial — important differential)

Why 3-class instead of binary:
  COVID-19 chest X-ray presentation (ground-glass opacities) closely
  resembles viral pneumonia. A binary NORMAL/COVID model would silently
  fail on non-COVID pneumonia cases. Including PNEUMONIA as an explicit
  third class lets the model express that distinction directly rather
  than forcing a false binary choice.

Dataset quality note:
  Public COVID-19 X-ray datasets are smaller and more heterogeneous
  (multiple hospital sources, inconsistent imaging protocols) than the
  Pneumonia/TB datasets. Expect lower ceiling accuracy and validate
  carefully — see training/covid19/train.py docstring for dataset
  curation notes.

Dataset:
  COVID-19 Radiography Database
  https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
  ~21,000 images across COVID/Normal/Viral Pneumonia/Lung Opacity
  (Lung Opacity class is merged into PNEUMONIA for this 3-class setup)
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from core.logger import logger
from core.config import settings


CLASS_NAMES  = ["NORMAL", "COVID-19", "PNEUMONIA"]
NUM_CLASSES  = len(CLASS_NAMES)
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class Covid19Classifier(nn.Module):
    """
    DenseNet-121 backbone — identical structure to PneumoniaClassifier
    and TuberculosisClassifier. Only the dataset and class set differ.
    """

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


def build_model() -> Covid19Classifier:
    model = Covid19Classifier()
    logger.info(f"Built Covid19Classifier | classes={CLASS_NAMES}")
    return model


def load_trained_model(
    model_path: Path | None = None,
    device: str | None = None,
) -> Covid19Classifier:
    device = device or settings.device
    model_path = model_path or (
        settings.model_registry_dir / "covid19" / "best_model.pth"
    )

    model = Covid19Classifier()

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded COVID-19 model from {model_path} on {device}")
    else:
        logger.warning(
            f"No trained COVID-19 model found at {model_path}. "
            "Using ImageNet backbone weights only — run training first."
        )

    model = model.to(device)
    model.eval()
    return model
