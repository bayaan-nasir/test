"""
tests/unit/test_pneumonia_inference.py
Unit tests for the pneumonia inference pipeline.

Run: pytest tests/ -v
"""
import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

from models.pneumonia.model import (
    PneumoniaClassifier, CLASS_NAMES, IDX_TO_CLASS, NUM_CLASSES
)
from preprocessing.image_transforms import preprocess_image_for_inference
from core.triage import compute_triage, TriageFlag


# Fixtures 
@pytest.fixture
def dummy_xray():
    """224x224 grayscale image simulating a chest X-ray."""
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def model():
    """Untrained model (ImageNet backbone weights only) for structural tests."""
    m = PneumoniaClassifier()
    m.eval()
    return m


# Model structure 
class TestModelArchitecture:
    def test_output_shape(self, model, dummy_xray):
        tensor = preprocess_image_for_inference(dummy_xray)
        with torch.no_grad():
            logits = model(tensor)
        assert logits.shape == (1, NUM_CLASSES), (
            f"Expected output shape (1, {NUM_CLASSES}), got {logits.shape}"
        )

    def test_output_num_classes(self, model, dummy_xray):
        tensor = preprocess_image_for_inference(dummy_xray)
        with torch.no_grad():
            logits = model(tensor)
        assert logits.shape[1] == 3, "Model must output 3 classes"

    def test_softmax_sums_to_one(self, model, dummy_xray):
        tensor = preprocess_image_for_inference(dummy_xray)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_class_names_match(self):
        assert CLASS_NAMES == ["NORMAL", "PNEUMONIA", "COVID-19"]
        assert len(IDX_TO_CLASS) == 3


# Preprocessing 
class TestPreprocessing:
    def test_output_tensor_shape(self, dummy_xray):
        tensor = preprocess_image_for_inference(dummy_xray)
        assert tensor.shape == (1, 3, 224, 224)

    def test_output_dtype(self, dummy_xray):
        tensor = preprocess_image_for_inference(dummy_xray)
        assert tensor.dtype == torch.float32

    def test_handles_grayscale_input(self):
        grayscale = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode="L")
        tensor = preprocess_image_for_inference(grayscale)
        assert tensor.shape == (1, 3, 224, 224), "Grayscale must be converted to 3-channel"


# Triage
class TestTriage:
    def test_high_confidence_severe_disease(self):
        flag, action = compute_triage("PNEUMONIA", 0.92)
        assert flag == TriageFlag.HIGH

    def test_high_confidence_normal(self):
        flag, action = compute_triage("NORMAL", 0.95)
        assert flag == TriageFlag.MEDIUM

    def test_medium_confidence_severe(self):
        flag, action = compute_triage("PNEUMONIA", 0.70)
        assert flag == TriageFlag.MEDIUM

    def test_low_confidence_always_low(self):
        flag, action = compute_triage("PNEUMONIA", 0.40)
        assert flag == TriageFlag.LOW

    def test_action_text_not_empty(self):
        _, action = compute_triage("PNEUMONIA", 0.88)
        assert len(action) > 10

    def test_covid_is_high_severity(self):
        flag, _ = compute_triage("COVID-19", 0.88)
        assert flag == TriageFlag.HIGH
