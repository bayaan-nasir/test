"""
core/router.py
Domain router — decides which models to invoke based on available inputs.

IMAGE ROUTING:
  xray        → [pneumonia, tuberculosis, covid19]
  skin        → [skin_cancer, skin_conditions]
  blood_smear → [malaria]
  histology   → [breast_cancer]

  When an image is present, ALL purely-tabular models that rely on
  the same domain AS the image type are suppressed. Cross-domain tabular
  models (diabetes, anaemia, hypertension, hepatitis_b, heart_disease)
  still run if the doctor provided those fields.

TABULAR ROUTING:
  Each tabular model has a minimum set of required fields. If fewer than
  the minimum are provided, the model is skipped with a clear reason.
  This prevents the model running on all-NaN input and producing garbage.

  MINIMUM FIELD THRESHOLDS:
    heart_disease : 5 of 13 cardiovascular fields
    diabetes      : 3 of 8 metabolic fields (glucose is strongly preferred)
    anaemia       : 2 of 5 CBC fields (hemoglobin strongly preferred)
    hypertension  : 3 of 8 lifestyle fields
    hepatitis_b   : 3 of 11 liver panel fields
"""
from dataclasses import dataclass, field
from api.schemas.unified import SymptomsInput, ImageType


@dataclass
class RoutingDecision:
    image_models:    list[str] = field(default_factory=list)
    tabular_models:  list[str] = field(default_factory=list)
    skipped:         dict[str, str] = field(default_factory=dict)  # model → reason


#Field membership definitions 
HEART_DISEASE_FIELDS   = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                           "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
HYPERTENSION_FIELDS    = ["age", "salt_intake", "bmi", "stress_score",
                           "sleep_hours", "smoking", "family_history", "physical_activity"]
DIABETES_FIELDS        = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                           "insulin", "bmi", "diabetes_pedigree_function", "age"]
DIABETES_SYMPTOMS_FIELDS = ["pregnancies", "glucose", "skin_thickness",
                             "insulin", "bmi", "diabetes_pedigree_function"]
ANAEMIA_FIELDS         = ["hemoglobin", "mch", "mchc", "mcv"]
HEPATITIS_FIELDS       = ["bilirubin", "alk_phosphate", "sgot", "albumin",
                           "protime", "fatigue", "malaise", "ascites", "varices"]


def _count_provided(data: SymptomsInput, field_list: list[str]) -> int:
    """Count how many fields from field_list have non-None values."""
    return sum(1 for f in field_list if getattr(data, f, None) is not None)


def route(
    symptoms: SymptomsInput | None,
    image_type: ImageType | None,
) -> RoutingDecision:
    """
    Determine which models to run given the available inputs.

    Args:
        symptoms:   SymptomsInput with optional fields (can be None for image-only)
        image_type: ImageType enum if an image was uploaded (None for symptoms-only)

    Returns:
        RoutingDecision with image_models, tabular_models, and skipped dict
    """
    decision = RoutingDecision()
    s = symptoms  # shorthand

    # 1. Image routing 
    if image_type is not None:
        if image_type == ImageType.XRAY:
            decision.image_models = ["pneumonia", "tuberculosis", "covid19"]

        elif image_type == ImageType.SKIN:
            decision.image_models = ["skin_cancer", "skin_conditions"]

        elif image_type == ImageType.BLOOD_SMEAR:
            decision.image_models = ["malaria"]

        elif image_type == ImageType.HISTOLOGY:
            decision.image_models = ["breast_cancer"]
    else:
        # No image — skip all image models
        for m in ["pneumonia", "tuberculosis", "covid19",
                  "malaria", "skin_cancer", "skin_conditions", "breast_cancer"]:
            decision.skipped[m] = "No image uploaded"

    # 2. Tabular routing 
    if s is None:
        for m in ["heart_disease", "diabetes", "anaemia", "hypertension", "hepatitis_b"]:
            decision.skipped[m] = "No symptom data provided"
        return decision

    # Heart Disease — needs cardiovascular data
    heart_count = _count_provided(s, HEART_DISEASE_FIELDS)
    if heart_count >= 5:
        decision.tabular_models.append("heart_disease")
    else:
        decision.skipped["heart_disease"] = (
            f"Insufficient cardiovascular data ({heart_count}/13 fields provided, need ≥5)"
        )

    # Hypertension — needs lifestyle data
    htn_count = _count_provided(s, HYPERTENSION_FIELDS)
    if htn_count >= 3:
        decision.tabular_models.append("hypertension")
    else:
        decision.skipped["hypertension"] = (
            f"Insufficient lifestyle data ({htn_count}/8 fields provided, need ≥3)"
        )

    # Diabetes — needs metabolic data; glucose is the most important single field
    diab_count = _count_provided(s, DIABETES_SYMPTOMS_FIELDS)
    has_glucose = s.glucose is not None
    if diab_count >= 3 or (diab_count >= 2 and has_glucose):
        decision.tabular_models.append("diabetes")
    else:
        decision.skipped["diabetes"] = (
            f"Insufficient metabolic data ({diab_count}/6 fields provided, need ≥3 or glucose+1 other)"
        )

    # Anaemia — needs CBC values; hemoglobin alone is actually meaningful
    anaemia_count = _count_provided(s, ANAEMIA_FIELDS)
    has_hgb = s.hemoglobin is not None
    if anaemia_count >= 2 or has_hgb:
        decision.tabular_models.append("anaemia")
    else:
        decision.skipped["anaemia"] = (
            f"Insufficient CBC data ({anaemia_count}/4 fields provided; "
            "provide at least hemoglobin)"
        )

    # Hepatitis B — needs liver panel data
    hep_count = _count_provided(s, HEPATITIS_FIELDS)
    if hep_count >= 3:
        decision.tabular_models.append("hepatitis_b")
    else:
        decision.skipped["hepatitis_b"] = (
            f"Insufficient liver panel data ({hep_count}/9 fields provided, need ≥3)"
        )

    return decision
