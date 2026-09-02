# ml_service/preprocessing/malaria_transforms.py
# Preprocessing and augmentation for malaria blood smear cell images.
# The NIH malaria dataset contains small, uniform images of single cells.
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

MALARIA_IMAGE_SIZE = 128   # NIH dataset cells are small — 128 is sufficient


def get_malaria_train_transforms() -> A.Compose:
    """
    Training augmentations for blood smear cell images.
    Stronger augmentation than X-rays — cells are small and uniform.
    """
    return A.Compose([
        A.Resize(MALARIA_IMAGE_SIZE, MALARIA_IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        # Gentle saturation shift — preserves Giemsa stain colour
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_malaria_val_transforms() -> A.Compose:
    """Validation / inference — resize and normalise only."""
    return A.Compose([
        A.Resize(MALARIA_IMAGE_SIZE, MALARIA_IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def preprocess_malaria_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL blood smear image for inference.

    Args:
        image: PIL.Image — the uploaded cell image

    Returns:
        torch.Tensor of shape (1, 3, 128, 128)
    """
    image = image.convert("RGB")
    transform = get_malaria_val_transforms()
    img_array = np.array(image)
    transformed = transform(image=img_array)
    tensor = transformed["image"]
    return tensor.unsqueeze(0)   # (1, 3, H, W)
