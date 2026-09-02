"""
tests/unit/test_skin_breast_cancer.py
Unit tests for the Skin Cancer (7-class) and Breast Cancer (binary) pipelines.

Run from ml_service root (Windows CMD):
  python -m pytest tests\\ -v
"""
import pytest
import torch
import numpy as np
from PIL import Image

from models.skin_cancer.model import (
    SkinCancerClassifier, CLASS_NAMES as SKIN_CLASSES, MALIGNANT_CLASSES
)
from preprocessing.skin_cancer_transforms import preprocess_skin_image, SKIN_IMAGE_SIZE

from models.breast_cancer.model import (
    BreastCancerClassifier, CLASS_NAMES as BREAST_CLASSES, BREAST_IMAGE_SIZE
)
from preprocessing.image_transforms import preprocess_image_for_inference

from core.triage import compute_triage, TriageFlag


# Skin Cancer
@pytest.fixture
def dummy_skin_lesion():
    arr = np.random.randint(80, 220, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def skin_model():
    m = SkinCancerClassifier()
    m.eval()
    return m


class TestSkinCancerModel:
    def test_output_shape(self, skin_model, dummy_skin_lesion):
        tensor = preprocess_skin_image(dummy_skin_lesion)
        with torch.no_grad():
            logits = skin_model(tensor)
        assert logits.shape == (1, 7), "Skin cancer must output 7 classes"

    def test_class_names_count(self):
        assert len(SKIN_CLASSES) == 7

    def test_melanoma_in_classes(self):
        assert "MELANOMA" in SKIN_CLASSES

    def test_malignant_classes_defined(self):
        assert "MELANOMA" in MALIGNANT_CLASSES
        assert "BASAL_CELL_CARCINOMA" in MALIGNANT_CLASSES

    def test_softmax_sums_to_one(self, skin_model, dummy_skin_lesion):
        tensor = preprocess_skin_image(dummy_skin_lesion)
        with torch.no_grad():
            probs = torch.softmax(skin_model(tensor), dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5


class TestSkinCancerPreprocessing:
    def test_output_shape(self, dummy_skin_lesion):
        tensor = preprocess_skin_image(dummy_skin_lesion)
        assert tensor.shape == (1, 3, SKIN_IMAGE_SIZE, SKIN_IMAGE_SIZE)

    def test_handles_grayscale(self):
        gray = Image.fromarray(np.random.randint(0, 255, (300, 300), dtype=np.uint8), mode="L")
        tensor = preprocess_skin_image(gray)
        assert tensor.shape[1] == 3


class TestSkinCancerTriage:
    def test_melanoma_high_confidence_is_high(self):
        flag, _ = compute_triage("MELANOMA", 0.88)
        assert flag == TriageFlag.HIGH

    def test_benign_nevus_low_confidence_is_low(self):
        flag, _ = compute_triage("MELANOCYTIC_NEVUS", 0.40)
        assert flag == TriageFlag.LOW


#  Breast Cancer 
@pytest.fixture
def dummy_histology_image():
    arr = np.random.randint(100, 230, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def breast_model():
    m = BreastCancerClassifier()
    m.eval()
    return m


class TestBreastCancerModel:
    def test_output_shape(self, breast_model, dummy_histology_image):
        tensor = preprocess_image_for_inference(dummy_histology_image)
        with torch.no_grad():
            logits = breast_model(tensor)
        assert logits.shape == (1, 2), "Breast cancer must output 2 classes"

    def test_class_names(self):
        assert BREAST_CLASSES == ["BENIGN", "MALIGNANT"]

    def test_softmax_sums_to_one(self, breast_model, dummy_histology_image):
        tensor = preprocess_image_for_inference(dummy_histology_image)
        with torch.no_grad():
            probs = torch.softmax(breast_model(tensor), dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_image_size_constant(self):
        assert BREAST_IMAGE_SIZE == 224


class TestBreastCancerTriage:
    def test_malignant_high_confidence_is_high(self):
        flag, _ = compute_triage("breast_cancer", 0.93)
        assert flag == TriageFlag.HIGH

    def test_benign_high_confidence_is_medium(self):
        flag, _ = compute_triage("NORMAL", 0.90)
        assert flag == TriageFlag.MEDIUM
