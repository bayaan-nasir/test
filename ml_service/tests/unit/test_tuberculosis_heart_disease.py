"""
tests/unit/test_tuberculosis_heart_disease.py
Unit tests for the Tuberculosis (image) and Heart Disease (tabular) pipelines.

Run from ml_service root (Windows CMD):
  python -m pytest tests\\ -v
"""
import pytest
import torch
import numpy as np
from PIL import Image

from models.tuberculosis.model import TuberculosisClassifier, CLASS_NAMES as TB_CLASSES
from preprocessing.image_transforms import preprocess_image_for_inference
from preprocessing.tabular_transforms import (
    HeartDiseaseInput, preprocess_heart_disease_input, HEART_DISEASE_FEATURE_ORDER
)
from core.triage import compute_triage, TriageFlag


#Tuberculosis 
@pytest.fixture
def dummy_xray():
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def tb_model():
    m = TuberculosisClassifier()
    m.eval()
    return m


class TestTuberculosisModel:
    def test_output_shape(self, tb_model, dummy_xray):
        tensor = preprocess_image_for_inference(dummy_xray)
        with torch.no_grad():
            logits = tb_model(tensor)
        assert logits.shape == (1, 2), "TB must output 2 classes (NORMAL, TUBERCULOSIS)"

    def test_class_names(self):
        assert TB_CLASSES == ["NORMAL", "TUBERCULOSIS"]

    def test_softmax_sums_to_one(self, tb_model, dummy_xray):
        tensor = preprocess_image_for_inference(dummy_xray)
        with torch.no_grad():
            probs = torch.softmax(tb_model(tensor), dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5


class TestTuberculosisTriage:
    def test_tb_high_confidence_is_high(self):
        flag, _ = compute_triage("tuberculosis", 0.90)
        assert flag == TriageFlag.HIGH

    def test_normal_high_confidence_is_medium(self):
        flag, _ = compute_triage("NORMAL", 0.92)
        assert flag == TriageFlag.MEDIUM


# Heart Disease
@pytest.fixture
def heart_patient_low_risk():
    return HeartDiseaseInput(
        age=35, sex=0, cp=0, trestbps=110, chol=180, fbs=0,
        restecg=0, thalach=170, exang=0, oldpeak=0.2, slope=2, ca=0, thal=1,
    )


@pytest.fixture
def heart_patient_high_risk():
    return HeartDiseaseInput(
        age=62, sex=1, cp=3, trestbps=150, chol=280, fbs=1,
        restecg=1, thalach=120, exang=1, oldpeak=2.5, slope=0, ca=2, thal=3,
    )


class TestHeartDiseaseInput:
    def test_valid_low_risk_patient(self, heart_patient_low_risk):
        assert heart_patient_low_risk.age == 35
        assert heart_patient_low_risk.cp == 0

    def test_invalid_sex_value(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            HeartDiseaseInput(
                age=40, sex=2, cp=0, trestbps=120, chol=200, fbs=0,
                restecg=0, thalach=150, exang=0, oldpeak=0, slope=1, ca=0, thal=1,
            )

    def test_feature_count(self):
        assert len(HEART_DISEASE_FEATURE_ORDER) == 13


class TestHeartDiseasePreprocessing:
    def test_output_shape(self, heart_patient_low_risk):
        X = preprocess_heart_disease_input(heart_patient_low_risk)
        assert X.shape == (1, 13)

    def test_output_dtype(self, heart_patient_low_risk):
        X = preprocess_heart_disease_input(heart_patient_low_risk)
        assert X.dtype == np.float32

    def test_values_preserved_in_order(self, heart_patient_high_risk):
        X = preprocess_heart_disease_input(heart_patient_high_risk)
        assert X[0, HEART_DISEASE_FEATURE_ORDER.index("age")] == 62.0
        assert X[0, HEART_DISEASE_FEATURE_ORDER.index("chol")] == 280.0


class TestHeartDiseaseTriage:
    def test_high_confidence_disease_is_high(self):
        flag, _ = compute_triage("heart_disease", 0.91)
        assert flag == TriageFlag.HIGH

    def test_low_confidence_is_low(self):
        flag, _ = compute_triage("heart_disease", 0.45)
        assert flag == TriageFlag.LOW
