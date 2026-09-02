"""
explainability/gradcam.py
Generates Grad-CAM heatmap overlays for image model predictions.

Grad-CAM highlights the regions of the X-ray the model focused on
when making its prediction — critical for clinician trust and audit.

Output: a PNG file with the original X-ray overlaid with a colour heatmap.
  - Red/yellow areas = regions most influential for the prediction
  - Blue areas = least influential regions
"""
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import uuid
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from core.config import settings
from core.logger import logger


class PneumoniaGradCAM:
    """
    Wraps pytorch-grad-cam for the PneumoniaClassifier.
    Target layer: the last convolutional block of DenseNet-121 (denseblock4).
    """

    def __init__(self, model: torch.nn.Module, device: str | None = None):
        self.model = model
        self.device = device or settings.device
        self.model.eval()

        # Target layer: last dense block of DenseNet-121
        # This is where the spatially meaningful feature maps live
        target_layer = model.features.denseblock4
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def generate(
        self,
        input_tensor: torch.Tensor,
        predicted_class_idx: int,
        original_image: Image.Image,
        save_dir: Path | None = None,
    ) -> tuple[str, np.ndarray]:
        """
        Generate and save a Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed tensor (1, 3, H, W) — from preprocess_image_for_inference
            predicted_class_idx: the class index the model predicted
            original_image: the original PIL image (before transforms)
            save_dir: directory to save the output PNG

        Returns:
            (relative_url, cam_array) — URL for the Django response, raw array for debugging
        """
        save_dir = save_dir or settings.gradcam_output_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        targets = [ClassifierOutputTarget(predicted_class_idx)]

        # Compute Grad-CAM
        grayscale_cam = self.cam(
            input_tensor=input_tensor.to(self.device),
            targets=targets
        )
        grayscale_cam = grayscale_cam[0]   # shape: (H, W)

        # Prepare original image as float RGB (0–1) for overlay
        original_rgb = original_image.convert("RGB").resize(
            (settings.image_size, settings.image_size)
        )
        original_np = np.array(original_rgb, dtype=np.float32) / 255.0

        # Overlay heatmap on original image
        cam_image = show_cam_on_image(
            original_np,
            grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
        )

        # Save output
        filename = f"gradcam_{uuid.uuid4().hex[:12]}.png"
        save_path = save_dir / filename
        Image.fromarray(cam_image).save(save_path)

        relative_url = f"/media/gradcam/{filename}"
        logger.debug(f"Grad-CAM saved → {save_path}")

        return relative_url, grayscale_cam
