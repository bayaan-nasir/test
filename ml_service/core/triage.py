"""
core/triage.py
Computes a triage flag (high / medium / low) for every prediction.
Based on confidence score + whether the predicted condition is acute/serious.
"""
from enum import Enum
from core.config import settings

# Diseases that should trigger HIGH triage even at medium confidence
HIGH_SEVERITY_CONDITIONS = {
    "pneumonia",
    "tuberculosis",
    "covid-19",
    "malaria",
    "cholera",
    "meningitis",
    "breast_cancer",
    "cervical_cancer",
    "heart_disease",
    "melanoma",
    "basal_cell_carcinoma",
    "hepatitis_b",
}


class TriageFlag(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


TRIAGE_ACTIONS = {
    TriageFlag.HIGH: "Seek immediate medical attention. This condition requires urgent clinical assessment.",
    TriageFlag.MEDIUM: "Consult a doctor soon. A follow-up examination is recommended within 24–48 hours.",
    TriageFlag.LOW: "Monitor your symptoms. If they persist or worsen, please consult a healthcare provider.",
}


def compute_triage(predicted_class: str, confidence: float) -> tuple[TriageFlag, str]:
    """
    Returns (triage_flag, recommended_action_text).

    Rules:
      - confidence >= HIGH threshold → HIGH if severe disease, else MEDIUM
      - confidence >= MEDIUM threshold → MEDIUM
      - confidence < MEDIUM threshold → LOW (model uncertain)
      - Severe diseases always escalate one level up
    """
    is_severe = predicted_class.lower().replace(" ", "_") in HIGH_SEVERITY_CONDITIONS

    if confidence >= settings.confidence_threshold_high:
        flag = TriageFlag.HIGH if is_severe else TriageFlag.MEDIUM
    elif confidence >= settings.confidence_threshold_medium:
        flag = TriageFlag.MEDIUM if is_severe else TriageFlag.LOW
    else:
        # Low confidence → always LOW regardless of disease
        flag = TriageFlag.LOW

    return flag, TRIAGE_ACTIONS[flag]
