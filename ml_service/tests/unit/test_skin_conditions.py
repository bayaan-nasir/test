"""
tests/unit/test_skin_conditions.py
Unit tests for the Skin Conditions classifier (Eczema, Ringworm,
Psoriasis, Acne, Normal Skin).

Run from ml_service root (Windows CMD):
  python -m pytest tests\\ -v
"""
import pytest
import torch
import numpy as np
from PIL import Image

from models.skin_conditions.model import (
    SkinConditionsClassifier, CLASS_NAMES, NUM_CLASSES
)
from preprocessing.skin_cancer_transforms import preprocess_skin_image, SKIN_IMAGE_SIZE
from core.triage import compute_triage, TriageFlag


@pytest.fixture
def dummy_skin_photo():
    arr = np.random.randint(90, 210, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def skin_conditions_model():
    m = SkinConditionsClassifier()
    m.eval()
    return m


class TestSkinConditionsModel:
    def test_output_shape(self, skin_conditions_model, dummy_skin_photo):
        tensor = preprocess_skin_image(dummy_skin_photo)
        with torch.no_grad():
            logits = skin_conditions_model(tensor)
        assert logits.shape == (1, 5), "Skin conditions model must output 5 classes"

    def test_class_names(self):
        assert CLASS_NAMES == ["ECZEMA", "RINGWORM", "PSORIASIS", "ACNE", "NORMAL_SKIN"]

    def test_num_classes_constant(self):
        assert NUM_CLASSES == 5

    def test_softmax_sums_to_one(self, skin_conditions_model, dummy_skin_photo):
        tensor = preprocess_skin_image(dummy_skin_photo)
        with torch.no_grad():
            probs = torch.softmax(skin_conditions_model(tensor), dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5


class TestSkinConditionsPreprocessing:
    def test_output_shape(self, dummy_skin_photo):
        tensor = preprocess_skin_image(dummy_skin_photo)
        assert tensor.shape == (1, 3, SKIN_IMAGE_SIZE, SKIN_IMAGE_SIZE)

    def test_reuses_same_image_size_as_skin_cancer(self):
        # Confirms both models share the same preprocessing pipeline,
        # which is intentional (same modality, same transforms module)
        assert SKIN_IMAGE_SIZE == 300


class TestSkinConditionsTriage:
    def test_eczema_moderate_confidence_is_medium(self):
        flag, _ = compute_triage("ECZEMA", 0.75)
        assert flag in (TriageFlag.MEDIUM, TriageFlag.LOW)

    def test_normal_skin_high_confidence_is_medium(self):
        flag, _ = compute_triage("NORMAL", 0.92)
        assert flag == TriageFlag.MEDIUM

    def test_ringworm_is_not_high_severity_by_default(self):
        # Ringworm/Eczema/Psoriasis/Acne are manageable dermatological
        # conditions, not acute emergencies — should not auto-escalate to HIGH
        flag, _ = compute_triage("RINGWORM", 0.80)
        assert flag != TriageFlag.HIGH
