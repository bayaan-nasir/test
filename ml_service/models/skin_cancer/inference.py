"""
models/skin_cancer/inference.py
Inference engine for the 7-class skin lesion classifier.

Input:  PIL Image of a skin lesion photo
Output: PredictionResponse with:
  - Top predicted class out of 7 possible lesion types
  - Confidence score
  - Top-2 differentials (multi-class — more clinically useful than binary)
  - Triage flag — escalated automatically if the prediction (or a close
    differential) is malignant/pre-malignant, regardless of confidence
  - Grad-CAM heatmap

Clinical note:
  Unlike Pneumonia/Malaria/TB (binary), this is a 7-way decision. A
  MELANOMA prediction at even moderate confidence should never be
  dismissed — the triage logic checks both the top prediction AND
  whether MELANOMA appears among the close differentials.
"""
import torch
import torch.nn.functional as F
from PIL import Image
import uuid

from models.skin_cancer.model import (
    load_trained_model, IDX_TO_CLASS, SkinCancerClassifier, MALIGNANT_CLASSES
)
from preprocessing.skin_cancer_transforms import preprocess_skin_image
from explainability.skin_cancer_gradcam import SkinCancerGradCAM
from core.config import settings
from core.triage import compute_triage, TriageFlag
from core.logger import logger
from api.schemas.prediction import PredictionResponse, ExplainabilityOutput

MODEL_VERSION = "1.0.0"

# Any differential above this probability is considered "clinically present"
DIFFERENTIAL_CONCERN_THRESHOLD = 0.15


class SkinCancerPredictor:
    """Singleton-style predictor for skin lesion classification."""

    def __init__(self):
        self.device = settings.device
        self.model: SkinCancerClassifier = load_trained_model(device=self.device)
        self.gradcam = SkinCancerGradCAM(model=self.model, device=self.device)
        logger.info("SkinCancerPredictor ready")

    def predict(self, image: Image.Image) -> PredictionResponse:
        input_tensor = preprocess_skin_image(image).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
        probs  = F.softmax(logits, dim=1).squeeze()   # (7,)

        top_idx = int(probs.argmax().item())
        confidence = float(probs[top_idx].item())
        predicted_class = IDX_TO_CLASS[top_idx]

        # Top-2 differentials by probability (excluding the top prediction)
        sorted_indices = probs.argsort(descending=True).tolist()
        differential_indices = [i for i in sorted_indices if i != top_idx][:2]
        differentials = [
            f"{IDX_TO_CLASS[i]} ({probs[i] * 100:.1f}%)" for i in differential_indices
        ]

        # ── Malignancy-aware triage ────────────────────────────────────────────
        # Standard triage based on confidence
        triage_flag, recommended_action = compute_triage(predicted_class, confidence)

        # Escalate: if any malignant class appears as a real differential
        # (above threshold), never let triage fall below MEDIUM
        concerning_malignant_differential = any(
            IDX_TO_CLASS[i] in MALIGNANT_CLASSES and probs[i].item() >= DIFFERENTIAL_CONCERN_THRESHOLD
            for i in differential_indices
        )
        if concerning_malignant_differential and triage_flag == TriageFlag.LOW:
            triage_flag = TriageFlag.MEDIUM
            recommended_action = (
                "A potentially serious skin condition cannot be ruled out. "
                "Please consult a dermatologist for an in-person examination."
            )

        # 5. Grad-CAM
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
