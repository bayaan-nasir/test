"""
tests/unit/test_malaria_inference.py
Unit tests for the malaria cell classifier pipeline.

Run: pytest tests/ -v
"""
import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch

from models.malaria.model import (
    MalariaClassifier, CLASS_NAMES, IDX_TO_CLASS, NUM_CLASSES
)
from preprocessing.malaria_transforms import preprocess_malaria_image, MALARIA_IMAGE_SIZE
from core.triage import compute_triage, TriageFlag


@pytest.fixture
def dummy_cell_image():
    """Simulated blood smear cell — small square image with Giemsa-like colours."""
    arr = np.random.randint(100, 220, (130, 130, 3), dtype=np.uint8)
    # Simulate a purple dot (Plasmodium stain) in the centre
    arr[55:75, 55:75] = [120, 60, 150]
    return Image.fromarray(arr)


@pytest.fixture
def model():
    m = MalariaClassifier()
    m.eval()
    return m


class TestMalariaModelArchitecture:
    def test_output_shape(self, model, dummy_cell_image):
        tensor = preprocess_malaria_image(dummy_cell_image)
        with torch.no_grad():
            logits = model(tensor)
        assert logits.shape == (1, NUM_CLASSES)

    def test_num_classes_is_two(self, model, dummy_cell_image):
        tensor = preprocess_malaria_image(dummy_cell_image)
        with torch.no_grad():
            logits = model(tensor)
        assert logits.shape[1] == 2, "Malaria is binary — must output 2 classes"

    def test_class_names(self):
        assert CLASS_NAMES == ["UNINFECTED", "PARASITIZED"]

    def test_softmax_sums_to_one(self, model, dummy_cell_image):
        tensor = preprocess_malaria_image(dummy_cell_image)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5


class TestMalariaPreprocessing:
    def test_output_shape(self, dummy_cell_image):
        tensor = preprocess_malaria_image(dummy_cell_image)
        assert tensor.shape == (1, 3, MALARIA_IMAGE_SIZE, MALARIA_IMAGE_SIZE)

    def test_output_dtype(self, dummy_cell_image):
        tensor = preprocess_malaria_image(dummy_cell_image)
        assert tensor.dtype == torch.float32

    def test_handles_large_image(self):
        """Large image (e.g. full slide) should be resized correctly."""
        large = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        tensor = preprocess_malaria_image(large)
        assert tensor.shape == (1, 3, MALARIA_IMAGE_SIZE, MALARIA_IMAGE_SIZE)

    def test_handles_grayscale(self):
        gray = Image.fromarray(
            np.random.randint(0, 255, (130, 130), dtype=np.uint8), mode="L"
        )
        tensor = preprocess_malaria_image(gray)
        assert tensor.shape[1] == 3, "Grayscale must be converted to RGB"


class TestMalariaTriage:
    def test_parasitized_high_confidence_is_high(self):
        flag, _ = compute_triage("malaria", 0.90)
        assert flag == TriageFlag.HIGH

    def test_uninfected_high_confidence_is_medium(self):
        flag, _ = compute_triage("NORMAL", 0.95)
        assert flag == TriageFlag.MEDIUM

    def test_low_confidence_is_always_low(self):
        flag, _ = compute_triage("malaria", 0.45)
        assert flag == TriageFlag.LOW

    def test_action_text_present(self):
        _, action = compute_triage("malaria", 0.88)
        assert isinstance(action, str) and len(action) > 0
