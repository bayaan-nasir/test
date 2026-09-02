"""
core/image_router.py
Classifies an uploaded medical image and routes it to the correct disease model.

A clinician uploads ONE image. The router inspects the image to determine
which category it belongs to, then routes to the appropriate model:

  Chest X-ray        → Pneumonia, Tuberculosis, or COVID-19
  Skin lesion photo  → Skin Cancer (HAM10000 classes) or Skin Conditions (Eczema/Ringworm)
  Blood smear        → Malaria
  Histology slide    → Breast Cancer

Routing strategy:
  The clinician selects an image_type hint from the request.
  This is the most reliable signal — automatically inferring modality
  from the pixel content alone is a separate and harder ML problem.
  The system trusts the clinician's label and routes accordingly.

  image_type options:
    "chest_xray"    → runs all 3 respiratory models, returns ranked results
    "skin"          → runs Skin Cancer + Skin Conditions, returns ranked
    "blood_smear"   → Malaria only
    "histology"     → Breast Cancer only
"""
from enum import Enum


class ImageType(str, Enum):
    CHEST_XRAY  = "chest_xray"
    SKIN        = "skin"
    BLOOD_SMEAR = "blood_smear"
    HISTOLOGY   = "histology"


# Maps image_type → list of (disease_name, predictor_attr_on_main)
IMAGE_ROUTE_MAP: dict[ImageType, list[tuple[str, str]]] = {
    ImageType.CHEST_XRAY: [
        ("Pneumonia",     "pneumonia_predictor"),
        ("Tuberculosis",  "tuberculosis_predictor"),
        ("COVID-19",      "covid19_predictor"),
    ],
    ImageType.SKIN: [
        ("Skin Cancer",      "skin_cancer_predictor"),
        ("Skin Conditions",  "skin_conditions_predictor"),
    ],
    ImageType.BLOOD_SMEAR: [
        ("Malaria", "malaria_predictor"),
    ],
    ImageType.HISTOLOGY: [
        ("Breast Cancer", "breast_cancer_predictor"),
    ],
}

IMAGE_TYPE_DESCRIPTIONS = {
    ImageType.CHEST_XRAY:  "Chest X-ray — screens for Pneumonia, Tuberculosis, COVID-19",
    ImageType.SKIN:        "Skin lesion photo — screens for Skin Cancer and Skin Conditions (Eczema, Ringworm, etc.)",
    ImageType.BLOOD_SMEAR: "Blood smear microscopy — screens for Malaria",
    ImageType.HISTOLOGY:   "Histology/biopsy slide — screens for Breast Cancer",
}
