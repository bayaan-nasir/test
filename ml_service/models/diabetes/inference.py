# Type 2 Diabetes Inference Engine
import numpy as np
import uuid


from models.diabetes.model import (
    load_trained_model, IDX_TO_CLASS, CLASS_NAMES, FEATURE_NAMES
)


from preprocessing.tabular_transforms import DiabetesInput, preprocess_diabetes_input
from explainability.shap_explainer import compute_shap_for_prediction
from core.config import settings
from core.triage import compute_triage
from core.logger import logger
from api.schemas.prediction import PredictionResponse, ExplainabilityOutput

MODEL_VERSION = "1.0.0"


class DiabetesPredictor:
    """
    Singleton-style predictor for Type 2 Diabetes.
    Loaded once at API startup alongside image model predictors.
    """

    def __init__(self):
        self.pipeline = load_trained_model()
        if self.pipeline is None:
            logger.warning(
                "DiabetesPredictor: no trained model found. "
                "Predictions will fail until training is complete."
            )
        logger.info("DiabetesPredictor ready")

    def predict(self, data: DiabetesInput) -> PredictionResponse:
        """
        Run full inference on a DiabetesInput.

        Args:
            data: validated DiabetesInput from API

        Returns:
            PredictionResponse
        """
        if self.pipeline is None:
            raise RuntimeError(
                "No trained diabetes model loaded. Run: "
                "python -m training.diabetes.train"
            )

        # 1. Preprocess — convert Pydantic model to (1, 8) numpy array
        X = preprocess_diabetes_input(data)

        # 2. Predict
        probs       = self.pipeline.predict_proba(X)[0]   # (2,)
        top_idx     = int(np.argmax(probs))
        confidence  = float(probs[top_idx])
        predicted_class = IDX_TO_CLASS[top_idx]

        # 3. Differentials — the other class with its probability
        other_idx    = 1 - top_idx
        differentials = [
            f"{IDX_TO_CLASS[other_idx]} ({probs[other_idx] * 100:.1f}%)"
        ]

        # 4. Triage
        triage_flag, recommended_action = compute_triage(
            "diabetes" if predicted_class == "DIABETIC" else "NORMAL",
            confidence
        )

        # 5. SHAP explainability
        top_features = compute_shap_for_prediction(
            self.pipeline, X, FEATURE_NAMES, top_n=5
        )

        # 6. Build response
        return PredictionResponse(
            prediction_id=uuid.uuid4().hex,
            domain="diabetes_metabolic",
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