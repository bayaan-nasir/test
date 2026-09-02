"""
preprocessing/skin_cancer_transforms.py
Image transforms for dermoscopic skin lesion images (HAM10000).

Differences from chest X-ray and malaria transforms:
  - Larger input size (300x300) — matches EfficientNet-B3's native resolution
  - Colour is highly diagnostic (asymmetric pigmentation, redness, blue-white
    veil) so colour augmentation must be conservative
  - Lesions can appear at any rotation/orientation on the skin — full
    rotation augmentation is safe and beneficial
  - Hair occlusion is a known artefact in dermoscopic images — slight blur
    augmentation helps the model generalise past hair strands
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

SKIN_IMAGE_SIZE = 300


def get_skin_train_transforms() -> A.Compose:
    """Training augmentations for dermoscopic skin lesion images."""
    return A.Compose([
        A.Resize(SKIN_IMAGE_SIZE, SKIN_IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.6),     # lesions have no canonical orientation
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4
        ),
        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=15,
            val_shift_limit=8,
            p=0.3
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),   # simulates hair/lens blur
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_skin_val_transforms() -> A.Compose:
    """Validation / inference — resize and normalise only."""
    return A.Compose([
        A.Resize(SKIN_IMAGE_SIZE, SKIN_IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def preprocess_skin_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL skin lesion image for inference.

    Args:
        image: PIL.Image — the uploaded skin lesion photo

    Returns:
        torch.Tensor of shape (1, 3, 300, 300)
    """
    image = image.convert("RGB")
    transform = get_skin_val_transforms()
    img_array = np.array(image)
    transformed = transform(image=img_array)
    return transformed["image"].unsqueeze(0)
