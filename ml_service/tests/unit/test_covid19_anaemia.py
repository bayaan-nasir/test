"""
tests/unit/test_covid19_anaemia.py
Unit tests for COVID-19 (image, 3-class) and Anaemia (tabular, binary).

Run from ml_service root (Windows CMD):
  python -m pytest tests\\ -v
"""
import pytest
import torch
import numpy as np
from PIL import Image

from models.covid19.model import Covid19Classifier, CLASS_NAMES as COVID_CLASSES
from preprocessing.image_transforms import preprocess_image_for_inference

from preprocessing.tabular_transforms import (
    AnaemiaInput, preprocess_anaemia_input, ANAEMIA_FEATURE_ORDER
)
from core.triage import compute_triage, TriageFlag


# COVID-19 
@pytest.fixture
def dummy_xray():
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def covid_model():
    m = Covid19Classifier()
    m.eval()
    return m


class TestCovid19Model:
    def test_output_shape(self, covid_model, dummy_xray):
        tensor = preprocess_image_for_inference(dummy_xray)
        with torch.no_grad():
            logits = covid_model(tensor)
        assert logits.shape == (1, 3), "COVID-19 model must output 3 classes"

    def test_class_names(self):
        assert COVID_CLASSES == ["NORMAL", "COVID-19", "PNEUMONIA"]

    def test_softmax_sums_to_one(self, covid_model, dummy_xray):
        tensor = preprocess_image_for_inference(dummy_xray)
        with torch.no_grad():
            probs = torch.softmax(covid_model(tensor), dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5


class TestCovid19Triage:
    def test_covid_high_confidence_is_high(self):
        flag, _ = compute_triage("covid-19", 0.90)
        assert flag == TriageFlag.HIGH

    def test_normal_high_confidence_is_medium(self):
        flag, _ = compute_triage("NORMAL", 0.92)
        assert flag == TriageFlag.MEDIUM

    def test_pneumonia_differential_high(self):
        flag, _ = compute_triage("pneumonia", 0.85)
        assert flag == TriageFlag.HIGH


# Anaemia
@pytest.fixture
def healthy_cbc():
    return AnaemiaInput(gender=0, hemoglobin=13.5, mch=28.0, mchc=33.0, mcv=88.0)


@pytest.fixture
def anaemic_cbc():
    return AnaemiaInput(gender=0, hemoglobin=8.2, mch=20.0, mchc=27.0, mcv=70.0)


class TestAnaemiaInput:
    def test_valid_healthy_cbc(self, healthy_cbc):
        assert healthy_cbc.hemoglobin == 13.5

    def test_invalid_hemoglobin_too_high(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AnaemiaInput(gender=0, hemoglobin=50, mch=28.0, mchc=33.0, mcv=88.0)

    def test_feature_count(self):
        assert len(ANAEMIA_FEATURE_ORDER) == 5


class TestAnaemiaPreprocessing:
    def test_output_shape(self, healthy_cbc):
        X = preprocess_anaemia_input(healthy_cbc)
        assert X.shape == (1, 5)

    def test_output_dtype(self, healthy_cbc):
        X = preprocess_anaemia_input(healthy_cbc)
        assert X.dtype == np.float32

    def test_values_preserved(self, anaemic_cbc):
        X = preprocess_anaemia_input(anaemic_cbc)
        assert X[0, ANAEMIA_FEATURE_ORDER.index("Hemoglobin")] == 8.2


class TestAnaemiaTriage:
    def test_anaemic_high_confidence_is_high(self):
        flag, _ = compute_triage("anaemia", 0.88)
        assert flag == TriageFlag.HIGH

    def test_low_confidence_is_low(self):
        flag, _ = compute_triage("anaemia", 0.50)
        assert flag == TriageFlag.LOW
