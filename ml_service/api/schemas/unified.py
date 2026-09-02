"""
api/schemas/unified.py
Request and response schemas for the two unified diagnostic endpoints.

POST /api/ml/diagnose/image-based
POST /api/ml/diagnose/symptoms-based
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from api.schemas.prediction import TriageLevel


class ImageType(str, Enum):
    XRAY        = "xray"         # chest X-ray  → Pneumonia, TB, COVID-19
    SKIN        = "skin"         # skin photo   → Skin Cancer, Skin Conditions
    BLOOD_SMEAR = "blood_smear"  # microscopy   → Malaria
    HISTOLOGY   = "histology"    # slide        → Breast Cancer


class SymptomsInput(BaseModel):
    """
    All structured clinical fields. Every field is optional.
    The router uses whichever fields are provided to decide which models to run.
    """
    # Demographics
    age:    Optional[float] = Field(None, ge=1,  le=120)
    sex:    Optional[int]   = Field(None, ge=0,  le=1,  description="1=male 0=female")

    # Cardiovascular (Heart Disease + Hypertension)
    cp:       Optional[int]   = Field(None, ge=0, le=3)
    trestbps: Optional[float] = Field(None, ge=60, le=250, description="Resting BP (mmHg)")
    chol:     Optional[float] = Field(None, ge=80, le=600, description="Cholesterol (mg/dl)")
    fbs:      Optional[int]   = Field(None, ge=0, le=1)
    restecg:  Optional[int]   = Field(None, ge=0, le=2)
    thalach:  Optional[float] = Field(None, ge=60, le=250)
    exang:    Optional[int]   = Field(None, ge=0, le=1)
    oldpeak:  Optional[float] = Field(None, ge=0, le=10)
    slope:    Optional[int]   = Field(None, ge=0, le=2)
    ca:       Optional[int]   = Field(None, ge=0, le=4)
    thal:     Optional[int]   = Field(None, ge=0, le=3)

    # Hypertension lifestyle
    salt_intake:       Optional[float] = Field(None, ge=0, le=20)
    stress_score:      Optional[float] = Field(None, ge=0, le=10)
    sleep_hours:       Optional[float] = Field(None, ge=0, le=14)
    smoking:           Optional[int]   = Field(None, ge=0, le=1)
    family_history:    Optional[int]   = Field(None, ge=0, le=1)
    physical_activity: Optional[float] = Field(None, ge=0, le=20)

    # Diabetes / metabolic
    pregnancies:                Optional[float] = Field(None, ge=0, le=20)
    glucose:                    Optional[float] = Field(None, ge=0, le=300)
    skin_thickness:             Optional[float] = Field(None, ge=0, le=100)
    insulin:                    Optional[float] = Field(None, ge=0, le=900)
    bmi:                        Optional[float] = Field(None, ge=0, le=80)
    diabetes_pedigree_function: Optional[float] = Field(None, ge=0, le=3)

    # Anaemia (CBC)
    hemoglobin: Optional[float] = Field(None, ge=2,  le=22)
    mch:        Optional[float] = Field(None, ge=10, le=50)
    mchc:       Optional[float] = Field(None, ge=15, le=45)
    mcv:        Optional[float] = Field(None, ge=50, le=130)

    # Hepatitis B (liver panel)
    bilirubin:     Optional[float] = Field(None, ge=0, le=10)
    alk_phosphate: Optional[float] = Field(None, ge=0, le=500)
    sgot:          Optional[float] = Field(None, ge=0, le=600)
    albumin:       Optional[float] = Field(None, ge=0, le=7)
    protime:       Optional[float] = Field(None, ge=0, le=100)
    fatigue:       Optional[int]   = Field(None, ge=0, le=1)
    malaise:       Optional[int]   = Field(None, ge=0, le=1)
    ascites:       Optional[int]   = Field(None, ge=0, le=1)
    varices:       Optional[int]   = Field(None, ge=0, le=1)


class ModelResult(BaseModel):
    disease:         str
    domain:          str
    predicted_class: str
    confidence:      float
    confidence_pct:  str
    triage:          TriageLevel
    explainability:  dict
    model_used:      str


class UnifiedDiagnosticResponse(BaseModel):
    request_id:       str
    models_run:       list[str]
    models_skipped:   list[str]
    results:          list[ModelResult]
    top_result:       ModelResult
    overall_triage:   TriageLevel
    clinical_summary: str
    gemini_used:      bool
    image_type_used:  Optional[str] = None
    clinical_notes:   Optional[str] = None
    disclaimer: str = (
        "AI-assisted decision support for qualified medical professionals only. "
        "All predictions must be verified by clinical judgement."
    )
