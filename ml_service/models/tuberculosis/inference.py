"""
models/tuberculosis/inference.py
Inference engine for TB. Identical pattern to PneumoniaPredictor —
reuses the generic PneumoniaGradCAM wrapper (it targets denseblock4,
which exists identically on any DenseNet-121-based model).
"""
import torch
import torch.nn.functional as F
from PIL import Image
import uuid

from models.tuberculosis.model import load_trained_model, IDX_TO_CLASS, TuberculosisClassifier
from preprocessing.image_transforms import preprocess_image_for_inference
from explainability.gradcam import PneumoniaGradCAM  # generic DenseNet-121 Grad-CAM — reused as-is
from core.config import settings
from core.triage import compute_triage
from core.logger import logger
from api.schemas.prediction import PredictionResponse, ExplainabilityOutput

MODEL_VERSION = "1.0.0"


class TuberculosisPredictor:
    """Singleton-style predictor for TB — mirrors PneumoniaPredictor exactly."""

    def __init__(self):
        self.device = settings.device
        self.model: TuberculosisClassifier = load_trained_model(device=self.device)
        self.gradcam = PneumoniaGradCAM(model=self.model, device=self.device)
        logger.info("TuberculosisPredictor ready")

    def predict(self, image: Image.Image) -> PredictionResponse:
        input_tensor = preprocess_image_for_inference(image).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze()
        top_idx = int(probs.argmax().item())
        confidence = float(probs[top_idx].item())
        predicted_class = IDX_TO_CLASS[top_idx]

        other_idx = 1 - top_idx
        differentials = [f"{IDX_TO_CLASS[other_idx]} ({probs[other_idx]*100:.1f}%)"]

        triage_flag, recommended_action = compute_triage(
            "tuberculosis" if predicted_class == "TUBERCULOSIS" else "NORMAL",
            confidence,
        )

        with torch.enable_grad():
            heatmap_url, _ = self.gradcam.generate(
                input_tensor=input_tensor,
                predicted_class_idx=top_idx,
                original_image=image,
            )

        return PredictionResponse(
            prediction_id=uuid.uuid4().hex,
            domain="respiratory",
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            confidence_pct=f"{confidence * 100:.1f}%",
            differentials=differentials,
            triage=triage_flag,
            recommended_action=recommended_action,
            explainability=ExplainabilityOutput(type="grad_cam", heatmap_url=heatmap_url),
            model_version=MODEL_VERSION,
        )
