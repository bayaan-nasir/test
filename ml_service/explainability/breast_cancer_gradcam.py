"""
explainability/breast_cancer_gradcam.py
Grad-CAM heatmap generator for the BreastCancerClassifier (ResNet-50).

Target layer: the last block of self.features (which is ResNet's layer4 —
the final residual block before global average pooling). This is the
standard Grad-CAM target for any ResNet architecture.

Clinical relevance:
  Highlights the cellular regions (e.g. irregular nuclei clusters, high
  mitotic density) the model associated with malignancy — useful for a
  pathologist reviewing the AI's reasoning alongside the slide.
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
from models.breast_cancer.model import BREAST_IMAGE_SIZE


class BreastCancerGradCAM:
    """Grad-CAM wrapper for ResNet-50 BreastCancerClassifier."""

    def __init__(self, model: torch.nn.Module, device: str | None = None):
        self.model  = model
        self.device = device or settings.device
        self.model.eval()

        # self.features is a Sequential ending in ResNet's layer4 block
        target_layer = model.features[-1]
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def generate(
        self,
        input_tensor: torch.Tensor,
        predicted_class_idx: int,
        original_image: Image.Image,
        save_dir: Path | None = None,
    ) -> tuple[str, np.ndarray]:
        save_dir = Path(save_dir or settings.gradcam_output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        targets = [ClassifierOutputTarget(predicted_class_idx)]
        grayscale_cam = self.cam(
            input_tensor=input_tensor.to(self.device),
            targets=targets
        )[0]

        original_rgb = original_image.convert("RGB").resize(
            (BREAST_IMAGE_SIZE, BREAST_IMAGE_SIZE)
        )
        original_np = np.array(original_rgb, dtype=np.float32) / 255.0

        cam_image = show_cam_on_image(
            original_np, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET
        )

        filename  = f"breast_gradcam_{uuid.uuid4().hex[:12]}.png"
        save_path = save_dir / filename
        Image.fromarray(cam_image).save(save_path)

        relative_url = f"/media/gradcam/{filename}"
        logger.debug(f"Breast cancer Grad-CAM saved → {save_path}")
        return relative_url, grayscale_cam
