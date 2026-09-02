"""
models/covid19/inference.py
Inference engine for COVID-19. Same pattern as TuberculosisPredictor —
reuses the generic PneumoniaGradCAM wrapper since both are DenseNet-121.
"""
import torch
import torch.nn.functional as F
from PIL import Image
import uuid

from models.covid19.model import load_trained_model, IDX_TO_CLASS, Covid19Classifier
from preprocessing.image_transforms import preprocess_image_for_inference
from explainability.gradcam import PneumoniaGradCAM
from core.config import settings
from core.triage import compute_triage
from core.logger import logger
from api.schemas.prediction import PredictionResponse, ExplainabilityOutput

MODEL_VERSION = "1.0.0"


class Covid19Predictor:
    """Singleton-style predictor for COVID-19 — 3-class chest X-ray model."""

    def __init__(self):
        self.device = settings.device
        self.model: Covid19Classifier = load_trained_model(device=self.device)
        self.gradcam = PneumoniaGradCAM(model=self.model, device=self.device)
        logger.info("Covid19Predictor ready")

    def predict(self, image: Image.Image) -> PredictionResponse:
        input_tensor = preprocess_image_for_inference(image).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze()
        top_idx = int(probs.argmax().item())
        confidence = float(probs[top_idx].item())
        predicted_class = IDX_TO_CLASS[top_idx]

        sorted_indices = probs.argsort(descending=True).tolist()
        differential_indices = [i for i in sorted_indices if i != top_idx][:2]
        differentials = [
            f"{IDX_TO_CLASS[i]} ({probs[i] * 100:.1f}%)" for i in differential_indices
        ]

        triage_flag, recommended_action = compute_triage(
            "covid-19" if predicted_class == "COVID-19" else
            ("pneumonia" if predicted_class == "PNEUMONIA" else "NORMAL"),
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
