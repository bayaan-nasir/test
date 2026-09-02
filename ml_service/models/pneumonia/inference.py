"""
models/pneumonia/inference.py
The main inference function for pneumonia prediction.

This is what the FastAPI route calls. It:
  1. Preprocesses the uploaded image
  2. Runs the model to get probabilities
  3. Computes triage
  4. Generates a Grad-CAM heatmap
  5. Returns a structured PredictionResponse

All model state is held in the PneumoniaPredictor singleton
loaded once at API startup — not on every request.
"""
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import uuid

from models.pneumonia.model import (
    load_trained_model, PneumoniaClassifier,
    IDX_TO_CLASS, CLASS_NAMES, NUM_CLASSES
)
from preprocessing.image_transforms import preprocess_image_for_inference
from explainability.gradcam import PneumoniaGradCAM
from core.config import settings
from core.triage import compute_triage
from core.logger import logger
from api.schemas.prediction import PredictionResponse, ExplainabilityOutput


MODEL_VERSION = "1.0.0"


class PneumoniaPredictor:
    """
    Singleton-style predictor loaded once at API startup.
    Holds the model and Grad-CAM wrapper in memory.
    """

    def __init__(self):
        self.device = settings.device
        self.model: PneumoniaClassifier = load_trained_model(device=self.device)
        self.gradcam = PneumoniaGradCAM(model=self.model, device=self.device)
        logger.info("PneumoniaPredictor ready")

    def predict(self, image: Image.Image) -> PredictionResponse:
        """
        Run full inference pipeline on a PIL image.

        Args:
            image: PIL.Image — the uploaded chest X-ray

        Returns:
            PredictionResponse — the structured output Django receives
        """
        # ── 1. Preprocess ─────────────────────────────────────────────────────
        input_tensor = preprocess_image_for_inference(image)
        input_tensor = input_tensor.to(self.device)

        # ── 2. Forward pass (no gradients needed here) ────────────────────────
        with torch.no_grad():
            logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze()
        probs_list = probs.cpu().tolist()

        top_idx = int(probs.argmax().item())
        confidence = float(probs[top_idx].item())
        predicted_class = IDX_TO_CLASS[top_idx]

        # ── 3. Differential diagnoses ─────────────────────────────────────────
        # Sort all classes by probability, return top-2 alternatives
        sorted_indices = probs.argsort(descending=True).tolist()
        differentials = [
            IDX_TO_CLASS[i] for i in sorted_indices if i != top_idx
        ][:2]

        # ── 4. Triage ─────────────────────────────────────────────────────────
        triage_flag, recommended_action = compute_triage(predicted_class, confidence)

        # ── 5. Grad-CAM ───────────────────────────────────────────────────────
        # Re-enable gradients for Grad-CAM computation
        with torch.enable_grad():
            heatmap_url, _ = self.gradcam.generate(
                input_tensor=input_tensor,
                predicted_class_idx=top_idx,
                original_image=image,
            )

        # ── 6. Build response ─────────────────────────────────────────────────
        prediction_id = uuid.uuid4().hex

        return PredictionResponse(
            prediction_id=prediction_id,
            domain="respiratory",
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            confidence_pct=f"{confidence * 100:.1f}%",
            differentials=differentials,
            triage=triage_flag,
            recommended_action=recommended_action,
            explainability=ExplainabilityOutput(
                type="grad_cam",
                heatmap_url=heatmap_url,
            ),
            model_version=MODEL_VERSION,
        )
