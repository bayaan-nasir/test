
### api/schemas/prediction.py
### Pydantic models for API request validation and response serialisation.
### Django will receive the PredictionResponse JSON from the ML service.

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class TriageLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExplainabilityOutput(BaseModel):
    type: str = Field(..., description="'grad_cam' for image models, 'shap' for tabular")
    heatmap_url: Optional[str] = Field(None, description="Relative URL to Grad-CAM image")
    top_features: Optional[list[dict]] = Field(
        None, description="SHAP top features list for tabular models"
    )


class PredictionResponse(BaseModel):
    prediction_id: str = Field(..., description="Unique ID — store this in Django DB")
    domain: str = Field(..., description="e.g. 'respiratory'")
    predicted_class: str = Field(..., description="e.g. 'Pneumonia'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence 0–1")
    confidence_pct: str = Field(..., description="e.g. '87.3%'")
    differentials: list[str] = Field(
        default_factory=list,
        description="2–3 alternative diagnoses ordered by probability"
    )
    triage: TriageLevel
    recommended_action: str = Field(..., description="Plain-language action for patient view")
    explainability: ExplainabilityOutput
    model_version: str = Field(..., description="Model version used for this prediction")
    disclaimer: str = Field(
        default=(
            "This prediction is AI-assisted and intended to support, not replace, "
            "clinical judgement. Always consult a qualified healthcare provider."
        )
    )


class HealthCheckResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    device: str
    version: str = "1.0.0"
