"""
tests/unit/test_diabetes_inference.py
Unit tests for the Type 2 Diabetes tabular inference pipeline.

Run from ml_service root (Windows):
  python -m pytest tests\\ -v
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from preprocessing.tabular_transforms import (
    DiabetesInput, preprocess_diabetes_input,
    FEATURE_NAMES, ZERO_AS_NAN_FEATURES
)
from core.triage import compute_triage, TriageFlag


# Fixtures 
@pytest.fixture
def healthy_patient():
    """A patient with healthy values — should predict NON-DIABETIC."""
    return DiabetesInput(
        pregnancies=1,
        glucose=85,
        blood_pressure=70,
        skin_thickness=20,
        insulin=80,
        bmi=22.5,
        diabetes_pedigree_function=0.2,
        age=25,
    )


@pytest.fixture
def diabetic_patient():
    """A patient with high-risk values — likely to predict DIABETIC."""
    return DiabetesInput(
        pregnancies=5,
        glucose=168,
        blood_pressure=90,
        skin_thickness=35,
        insulin=0,    # missing — will be imputed
        bmi=39.5,
        diabetes_pedigree_function=1.1,
        age=52,
    )


# Input validation 
class TestDiabetesInputValidation:
    def test_valid_healthy_patient(self, healthy_patient):
        assert healthy_patient.glucose == 85
        assert healthy_patient.bmi == 22.5

    def test_zero_glucose_accepted(self):
        """Zero values are accepted — they get imputed downstream."""
        data = DiabetesInput(
            pregnancies=0, glucose=0, blood_pressure=0,
            skin_thickness=0, insulin=0, bmi=0,
            diabetes_pedigree_function=0.1, age=30
        )
        assert data.glucose == 0

    def test_invalid_glucose_too_high(self):
        """Glucose > 300 should raise a validation error."""
        import pytest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DiabetesInput(
                pregnancies=1, glucose=500, blood_pressure=70,
                skin_thickness=20, insulin=80, bmi=22.5,
                diabetes_pedigree_function=0.2, age=25
            )

    def test_all_features_present(self, healthy_patient):
        X = preprocess_diabetes_input(healthy_patient)
        assert X.shape == (1, len(FEATURE_NAMES))

    def test_correct_feature_count(self):
        assert len(FEATURE_NAMES) == 8


# Preprocessing 
class TestDiabetesPreprocessing:
    def test_output_shape(self, healthy_patient):
        X = preprocess_diabetes_input(healthy_patient)
        assert X.shape == (1, 8)

    def test_output_dtype(self, healthy_patient):
        X = preprocess_diabetes_input(healthy_patient)
        assert X.dtype == np.float32

    def test_zero_glucose_becomes_nan(self):
        """Zero glucose must be converted to NaN for imputation."""
        data = DiabetesInput(
            pregnancies=1, glucose=0, blood_pressure=70,
            skin_thickness=20, insulin=80, bmi=22.5,
            diabetes_pedigree_function=0.2, age=25
        )
        X = preprocess_diabetes_input(data)
        glucose_idx = FEATURE_NAMES.index("Glucose")
        assert np.isnan(X[0, glucose_idx]), "Zero glucose should become NaN"

    def test_zero_bmi_becomes_nan(self):
        data = DiabetesInput(
            pregnancies=1, glucose=85, blood_pressure=70,
            skin_thickness=20, insulin=80, bmi=0,
            diabetes_pedigree_function=0.2, age=25
        )
        X = preprocess_diabetes_input(data)
        bmi_idx = FEATURE_NAMES.index("BMI")
        assert np.isnan(X[0, bmi_idx]), "Zero BMI should become NaN"

    def test_nonzero_values_preserved(self, healthy_patient):
        X = preprocess_diabetes_input(healthy_patient)
        assert X[0, FEATURE_NAMES.index("Glucose")] == 85.0
        assert X[0, FEATURE_NAMES.index("Age")] == 25.0


# Triage 
class TestDiabetesTriage:
    def test_diabetic_high_confidence(self):
        flag, action = compute_triage("diabetes", 0.91)
        assert flag == TriageFlag.HIGH

    def test_non_diabetic_medium_confidence(self):
        flag, action = compute_triage("NORMAL", 0.88)
        assert flag == TriageFlag.MEDIUM

    def test_low_confidence_is_low(self):
        flag, action = compute_triage("diabetes", 0.50)
        assert flag == TriageFlag.LOW

    def test_action_text_exists(self):
        _, action = compute_triage("diabetes", 0.88)
        assert len(action) > 10
