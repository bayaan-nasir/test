"""
core/symptom_router.py
Maps the clinician's symptom submission to the relevant tabular disease models.

When no image is uploaded, the clinician submits a structured symptom form.
The router uses the reported symptoms + vital signs to determine which
tabular models are clinically relevant to run, then queries all of them
and returns a ranked multi-disease assessment.

Routing logic:
  Each disease has a set of "trigger features" — symptoms/lab values
  that, when present or abnormal, make that disease worth screening for.
  The router checks which triggers fire and runs only those models,
  avoiding irrelevant predictions (e.g. not screening for anaemia when
  the patient only has chest pain and shortness of breath).

  A model is ALWAYS run if its trigger score >= MIN_TRIGGER_SCORE.
  A model is ALWAYS skipped if its trigger score == 0.

  This is clinical pre-filtering, not a diagnosis. All triggered models
  run and the clinician sees all results.
"""
from dataclasses import dataclass, field
from typing import Any


MIN_TRIGGER_SCORE = 1  # at least 1 trigger must fire to run a model


@dataclass
class ModelTrigger:
    """Defines when a tabular model should be included in a prediction run."""
    disease:       str
    predictor_key: str          # attribute name on main module
    triggers:      list[str]    # symptom/field names that activate this model
    description:   str


# All tabular (structured data) models with their clinical triggers
TABULAR_MODELS: list[ModelTrigger] = [
    ModelTrigger(
        disease="Diabetes",
        predictor_key="diabetes_predictor",
        triggers=["glucose", "bmi", "insulin", "diabetes_pedigree_function",
                  "frequent_urination", "excessive_thirst", "fatigue"],
        description="Type 2 Diabetes — triggered by glucose, BMI, insulin, or metabolic symptoms",
    ),
    ModelTrigger(
        disease="Heart Disease",
        predictor_key="heart_disease_predictor",
        triggers=["chest_pain", "cp", "thalach", "exang", "oldpeak",
                  "chol", "trestbps", "shortness_of_breath"],
        description="Heart Disease — triggered by chest pain, ECG findings, or cardiovascular risk",
    ),
    ModelTrigger(
        disease="Anaemia",
        predictor_key="anaemia_predictor",
        triggers=["hemoglobin", "mch", "mchc", "mcv",
                  "fatigue", "pallor", "weakness"],
        description="Anaemia — triggered by CBC values or fatigue/pallor symptoms",
    ),
    ModelTrigger(
        disease="Hypertension",
        predictor_key="hypertension_predictor",
        triggers=["blood_pressure", "trestbps", "salt_intake", "bmi",
                  "stress_score", "family_history", "smoking",
                  "headache", "dizziness"],
        description="Hypertension — triggered by blood pressure, lifestyle risk factors",
    ),
    ModelTrigger(
        disease="Hepatitis B",
        predictor_key="hepatitis_b_predictor",
        triggers=["bilirubin", "sgot", "alk_phosphate", "albumin",
                  "jaundice", "abdominal_pain", "fatigue", "ascites",
                  "varices", "malaise"],
        description="Hepatitis B — triggered by liver enzyme values or hepatic symptoms",
    ),
]


def get_triggered_models(symptom_data: dict[str, Any]) -> list[ModelTrigger]:
    """
    Given the clinician's submitted symptom/lab data, return the list of
    tabular models that should be run.

    A model is triggered if at least one of its trigger fields:
      - Is present and non-None in the submitted data, OR
      - Is a boolean flag set to True

    Args:
        symptom_data: dict of field_name -> value from the unified request

    Returns:
        List of ModelTrigger instances for models that should run
    """
    triggered = []
    submitted_keys = {
        k for k, v in symptom_data.items()
        if v is not None and v is not False and v != 0
    }

    for model in TABULAR_MODELS:
        trigger_count = sum(
            1 for t in model.triggers if t in submitted_keys
        )
        if trigger_count >= MIN_TRIGGER_SCORE:
            triggered.append(model)

    return triggered


def get_all_tabular_models() -> list[ModelTrigger]:
    """Return all tabular models — used when the clinician explicitly
    requests a full panel screen."""
    return TABULAR_MODELS
