"""
models/hepatitis_b/inference.py
Inference engine for Hepatitis B. Mirrors DiabetesPredictor's pattern
(zero-as-missing handling) combined with the standard tabular SHAP flow.
"""
import numpy as np
import uuid

from models.hepatitis_b.model import load_trained_model, IDX_TO_CLASS
from preprocessing.tabular_transforms import (
    HepatitisBInput, preprocess_hepatitis_b_input, HEPATITIS_B_FEATURE_ORDER
)
from explainability.shap_explainer import compute_shap_for_prediction
from core.triage import compute_triage
from core.logger import logger
from api.schemas.prediction import PredictionResponse, ExplainabilityOutput

MODEL_VERSION = "1.0.0"


class HepatitisBPredictor:
    """Singleton-style predictor for Hepatitis B."""

    def __init__(self):
        self.pipeline = load_trained_model()
        if self.pipeline is None:
            logger.warning(
                "HepatitisBPredictor: no trained model found. "
                "Predictions will fail until training is complete."
            )
        logger.info("HepatitisBPredictor ready")

    def predict(self, data: HepatitisBInput) -> PredictionResponse:
        if self.pipeline is None:
            raise RuntimeError(
                "No trained hepatitis B model loaded. Run: "
                "python -m training.hepatitis_b.train"
            )

        X = preprocess_hepatitis_b_input(data)

        probs = self.pipeline.predict_proba(X)[0]
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        predicted_class = IDX_TO_CLASS[top_idx]

        other_idx = 1 - top_idx
        differentials = [f"{IDX_TO_CLASS[other_idx]} ({probs[other_idx] * 100:.1f}%)"]

        triage_flag, recommended_action = compute_triage(
            "hepatitis_b" if predicted_class == "HIGH RISK" else "NORMAL",
            confidence
        )

        top_features = compute_shap_for_prediction(
            self.pipeline, X, HEPATITIS_B_FEATURE_ORDER, top_n=5
        )

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
                type="shap", heatmap_url=None, top_features=top_features,
            ),
            model_version=MODEL_VERSION,
        )
