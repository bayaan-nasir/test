# ml_service/preprocessing/image_transforms.py
# Image preprocessing and augmentation for chest X-ray images.
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch
from core.config import settings

# ImageNet mean and std — required for pretrained DenseNet-121
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int | None = None) -> A.Compose:
    """
    Training augmentations for chest X-rays.
    Conservative augmentations — X-rays are sensitive images and
    aggressive transforms (heavy colour jitter, elastic deformations) 
    can corrupt diagnostically meaningful features.
    """
    size = image_size or settings.image_size
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            p=0.4
        ),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int | None = None) -> A.Compose:
    """
    Validation / inference transforms — resize and normalise only.
    No augmentation to ensure reproducible predictions.
    """
    size = image_size or settings.image_size
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def preprocess_image_for_inference(image: Image.Image) -> torch.Tensor:
    """
    Takes a PIL Image, applies val_transforms, returns a (1, 3, H, W) tensor.
    This is what the inference endpoint calls.

    Args:
        image: PIL.Image.Image — the uploaded chest X-ray

    Returns:
        torch.Tensor of shape (1, 3, image_size, image_size)
    """
    # Convert to RGB (handles grayscale DICOM-derived images)
    image = image.convert("RGB")

    transform = get_val_transforms()
    img_array = np.array(image)
    transformed = transform(image=img_array)
    tensor = transformed["image"]          # shape: (3, H, W)
    return tensor.unsqueeze(0)             # shape: (1, 3, H, W)
