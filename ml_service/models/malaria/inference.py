"""
models/malaria/inference.py
Inference engine for the malaria cell classifier.

Input:  PIL Image of a blood smear microscopy cell
Output: PredictionResponse with:
  - PARASITIZED or UNINFECTED prediction
  - Confidence score
  - Triage flag (PARASITIZED → always HIGH — malaria is a medical emergency)
  - Grad-CAM heatmap highlighting the Plasmodium parasite region

Clinical note:
  Malaria (Plasmodium falciparum) is one of Ghana's leading causes of
  morbidity and mortality. A PARASITIZED prediction should always trigger
  HIGH triage regardless of confidence — early treatment is critical.
"""
import torch
import torch.nn.functional as F
from PIL import Image
import uuid

from models.malaria.model import load_trained_model, IDX_TO_CLASS, MalariaClassifier
from preprocessing.malaria_transforms import preprocess_malaria_image
from explainability.malaria_gradcam import MalariaGradCAM
from core.config import settings
from core.triage import compute_triage
from core.logger import logger
from api.schemas.prediction import PredictionResponse, ExplainabilityOutput

MODEL_VERSION = "1.0.0"


class MalariaPredictor:
    """
    Singleton-style predictor loaded once at API startup.
    Mirrors the pattern established by PneumoniaPredictor.
    """

    def __init__(self):
        self.device  = settings.device
        self.model: MalariaClassifier = load_trained_model(device=self.device)
        self.gradcam = MalariaGradCAM(model=self.model, device=self.device)
        logger.info("MalariaPredictor ready")

    def predict(self, image: Image.Image) -> PredictionResponse:
        """
        Run full inference on a PIL blood smear cell image.

        Args:
            image: PIL.Image — uploaded microscopy cell image

        Returns:
            PredictionResponse
        """
        # 1. Preprocess
        input_tensor = preprocess_malaria_image(image).to(self.device)

        # 2. Forward pass
        with torch.no_grad():
            logits = self.model(input_tensor)
        probs  = F.softmax(logits, dim=1).squeeze()   # (2,)
        probs_list = probs.cpu().tolist()

        top_idx         = int(probs.argmax().item())
        confidence      = float(probs[top_idx].item())
        predicted_class = IDX_TO_CLASS[top_idx]

        # 3. Differential (only 2 classes — one differential)
        other_idx    = 1 - top_idx
        differentials = [IDX_TO_CLASS[other_idx]]

        # 4. Triage — PARASITIZED is always treated as high severity
        triage_flag, recommended_action = compute_triage(
            "malaria" if predicted_class == "PARASITIZED" else "NORMAL",
            confidence,
        )

        # 5. Grad-CAM
        with torch.enable_grad():
            heatmap_url, _ = self.gradcam.generate(
                input_tensor=input_tensor,
                predicted_class_idx=top_idx,
                original_image=image,
            )

        # 6. Build response
        return PredictionResponse(
            prediction_id=uuid.uuid4().hex,
            domain="infectious_diseases",
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
