"""
models/skin_conditions/inference.py
Inference engine for the 5-class skin conditions classifier
(Eczema, Ringworm, Psoriasis, Acne, Normal Skin).

Reuses the SkinCancerGradCAM class directly — it's architecture-generic
for any EfficientNet-B3 model with a `.features` Sequential container,
so no new Grad-CAM file is needed.
"""
import torch
import torch.nn.functional as F
from PIL import Image
import uuid

from models.skin_conditions.model import (
    load_trained_model, IDX_TO_CLASS, SkinConditionsClassifier
)
from preprocessing.skin_cancer_transforms import preprocess_skin_image
from explainability.skin_cancer_gradcam import SkinCancerGradCAM
from core.config import settings
from core.triage import compute_triage
from core.logger import logger
from api.schemas.prediction import PredictionResponse, ExplainabilityOutput

MODEL_VERSION = "1.0.0"


class SkinConditionsPredictor:
    """Singleton-style predictor for general skin conditions."""

    def __init__(self):
        self.device = settings.device
        self.model: SkinConditionsClassifier = load_trained_model(device=self.device)
        # Reuses the EfficientNet-B3 Grad-CAM wrapper built for Skin Cancer —
        # it targets model.features[-1], which exists identically here
        self.gradcam = SkinCancerGradCAM(model=self.model, device=self.device)
        logger.info("SkinConditionsPredictor ready")

    def predict(self, image: Image.Image) -> PredictionResponse:
        input_tensor = preprocess_skin_image(image).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
        probs  = F.softmax(logits, dim=1).squeeze()

        top_idx = int(probs.argmax().item())
        confidence = float(probs[top_idx].item())
        predicted_class = IDX_TO_CLASS[top_idx]

        sorted_indices = probs.argsort(descending=True).tolist()
        differential_indices = [i for i in sorted_indices if i != top_idx][:2]
        differentials = [
            f"{IDX_TO_CLASS[i]} ({probs[i] * 100:.1f}%)" for i in differential_indices
        ]

        # NORMAL_SKIN maps to a "no condition" triage baseline;
        # all other classes are mild/manageable dermatological conditions
        triage_flag, recommended_action = compute_triage(
            "NORMAL" if predicted_class == "NORMAL_SKIN" else predicted_class,
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
            domain="dermatology",
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            confidence_pct=f"{confidence * 100:.1f}%",
            differentials=differentials,
            triage=triage_flag,
            recommended_action=recommended_action,
            explainability=ExplainabilityOutput(type="grad_cam", heatmap_url=heatmap_url),
            model_version=MODEL_VERSION,
        )
