"""
models/heart_disease/inference.py
Inference engine for Heart Disease. Mirrors DiabetesPredictor exactly —
JSON input, SHAP explainability, no GPU, near-instant inference.
"""
import numpy as np
import uuid

from models.heart_disease.model import load_trained_model, IDX_TO_CLASS
from preprocessing.tabular_transforms import (
    HeartDiseaseInput, preprocess_heart_disease_input, HEART_DISEASE_FEATURE_ORDER
)
from explainability.shap_explainer import compute_shap_for_prediction
from core.triage import compute_triage
from core.logger import logger
from api.schemas.prediction import PredictionResponse, ExplainabilityOutput

MODEL_VERSION = "1.0.0"


class HeartDiseasePredictor:
    """Singleton-style predictor for Heart Disease — mirrors DiabetesPredictor."""

    def __init__(self):
        self.pipeline = load_trained_model()
        if self.pipeline is None:
            logger.warning(
                "HeartDiseasePredictor: no trained model found. "
                "Predictions will fail until training is complete."
            )
        logger.info("HeartDiseasePredictor ready")

    def predict(self, data: HeartDiseaseInput) -> PredictionResponse:
        if self.pipeline is None:
            raise RuntimeError(
                "No trained heart disease model loaded. Run: "
                "python -m training.heart_disease.train"
            )

        X = preprocess_heart_disease_input(data)

        probs = self.pipeline.predict_proba(X)[0]
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        predicted_class = IDX_TO_CLASS[top_idx]

        other_idx = 1 - top_idx
        differentials = [f"{IDX_TO_CLASS[other_idx]} ({probs[other_idx] * 100:.1f}%)"]

        triage_flag, recommended_action = compute_triage(
            "heart_disease" if predicted_class == "HEART DISEASE" else "NORMAL",
            confidence
        )

        top_features = compute_shap_for_prediction(
            self.pipeline, X, HEART_DISEASE_FEATURE_ORDER, top_n=5
        )

        return PredictionResponse(
            prediction_id=uuid.uuid4().hex,
            domain="cardiovascular",
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            confidence_pct=f"{confidence * 100:.1f}%",
            differentials=differentials,
            triage=triage_flag,
            recommended_action=recommended_action,
            explainability=ExplainabilityOutput(
                type="shap",
                heatmap_url=None,
                top_features=top_features,
            ),
            model_version=MODEL_VERSION,
        )
