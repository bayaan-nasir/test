"""
explainability/malaria_gradcam.py
Grad-CAM heatmap generator for the MalariaClassifier (EfficientNet-B0).

Target layer: the last convolutional block of EfficientNet-B0 (features[-1]).
This gives the most semantically meaningful spatial activations —
highlighting which parts of the blood smear cell the model focused on.

For malaria this is clinically meaningful:
  - PARASITIZED: model should focus on the dark Plasmodium ring/trophozoite
    stained by Giemsa — visible as a purple dot inside the red cell
  - UNINFECTED: model should show diffuse / no focus area
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
from preprocessing.malaria_transforms import MALARIA_IMAGE_SIZE


class MalariaGradCAM:
    """Grad-CAM wrapper for EfficientNet-B0 MalariaClassifier."""

    def __init__(self, model: torch.nn.Module, device: str | None = None):
        self.model  = model
        self.device = device or settings.device
        self.model.eval()

        # EfficientNet-B0: last block in self.features is features[-1]
        # This is the MBConv block with the richest spatial features
        target_layer = model.features[-1]
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def generate(
        self,
        input_tensor: torch.Tensor,
        predicted_class_idx: int,
        original_image: Image.Image,
        save_dir: Path | None = None,
    ) -> tuple[str, np.ndarray]:
        """
        Generate a Grad-CAM overlay for a blood smear cell image.

        Returns:
            (relative_url, grayscale_cam_array)
        """
        save_dir = Path(save_dir or settings.gradcam_output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        targets        = [ClassifierOutputTarget(predicted_class_idx)]
        grayscale_cam  = self.cam(
            input_tensor=input_tensor.to(self.device),
            targets=targets
        )[0]

        original_rgb = original_image.convert("RGB").resize(
            (MALARIA_IMAGE_SIZE, MALARIA_IMAGE_SIZE)
        )
        original_np = np.array(original_rgb, dtype=np.float32) / 255.0

        cam_image = show_cam_on_image(
            original_np,
            grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_INFERNO,  # warm colormap — clearer on small cells
        )

        filename     = f"malaria_gradcam_{uuid.uuid4().hex[:12]}.png"
        save_path    = save_dir / filename
        Image.fromarray(cam_image).save(save_path)

        relative_url = f"/media/gradcam/{filename}"
        logger.debug(f"Malaria Grad-CAM saved → {save_path}")
        return relative_url, grayscale_cam
