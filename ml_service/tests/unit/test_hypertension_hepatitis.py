"""
tests/unit/test_hypertension_hepatitis.py
Unit tests for Hypertension and Hepatitis B (both tabular, binary).

Run from ml_service root (Windows CMD):
  python -m pytest tests\\ -v
"""
import pytest
import numpy as np

from preprocessing.tabular_transforms import (
    HypertensionInput, preprocess_hypertension_input, HYPERTENSION_FEATURE_ORDER,
    HepatitisBInput, preprocess_hepatitis_b_input, HEPATITIS_B_FEATURE_ORDER,
)
from core.triage import compute_triage, TriageFlag


#Hypertension 
@pytest.fixture
def low_risk_patient():
    return HypertensionInput(
        age=28, salt_intake=4.0, bmi=22.0, stress_score=2,
        sleep_hours=8, smoking=0, family_history=0, physical_activity=6,
    )


@pytest.fixture
def high_risk_patient():
    return HypertensionInput(
        age=58, salt_intake=14.0, bmi=33.0, stress_score=8,
        sleep_hours=4, smoking=1, family_history=1, physical_activity=0.5,
    )


class TestHypertensionInput:
    def test_valid_low_risk(self, low_risk_patient):
        assert low_risk_patient.age == 28

    def test_invalid_smoking_value(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            HypertensionInput(
                age=40, salt_intake=5, bmi=25, stress_score=3,
                sleep_hours=7, smoking=2, family_history=0, physical_activity=3,
            )

    def test_feature_count(self):
        assert len(HYPERTENSION_FEATURE_ORDER) == 8


class TestHypertensionPreprocessing:
    def test_output_shape(self, low_risk_patient):
        X = preprocess_hypertension_input(low_risk_patient)
        assert X.shape == (1, 8)

    def test_values_preserved(self, high_risk_patient):
        X = preprocess_hypertension_input(high_risk_patient)
        assert X[0, HYPERTENSION_FEATURE_ORDER.index("bmi")] == 33.0
        assert X[0, HYPERTENSION_FEATURE_ORDER.index("smoking")] == 1.0


class TestHypertensionTriage:
    def test_hypertension_high_confidence_is_medium_not_high(self):
        # Hypertension is intentionally NOT in HIGH_SEVERITY_CONDITIONS —
        # it's a chronic manageable condition, not an acute emergency
        flag, _ = compute_triage("hypertension", 0.90)
        assert flag == TriageFlag.MEDIUM

    def test_no_hypertension_high_confidence_is_medium(self):
        flag, _ = compute_triage("NORMAL", 0.92)
        assert flag == TriageFlag.MEDIUM


# Hepatitis B
@pytest.fixture
def low_risk_liver():
    return HepatitisBInput(
        age=30, sex=0, bilirubin=0.8, alk_phosphate=80, sgot=25,
        albumin=4.2, protime=12, fatigue=0, malaise=0, ascites=0, varices=0,
    )


@pytest.fixture
def high_risk_liver():
    return HepatitisBInput(
        age=55, sex=1, bilirubin=3.5, alk_phosphate=210, sgot=180,
        albumin=2.6, protime=22, fatigue=1, malaise=1, ascites=1, varices=1,
    )


class TestHepatitisBInput:
    def test_valid_low_risk(self, low_risk_liver):
        assert low_risk_liver.bilirubin == 0.8

    def test_invalid_sex_value(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            HepatitisBInput(
                age=40, sex=3, bilirubin=1.0, alk_phosphate=90, sgot=30,
                albumin=4.0, protime=12, fatigue=0, malaise=0, ascites=0, varices=0,
            )

    def test_feature_count(self):
        assert len(HEPATITIS_B_FEATURE_ORDER) == 11


class TestHepatitisBPreprocessing:
    def test_output_shape(self, low_risk_liver):
        X = preprocess_hepatitis_b_input(low_risk_liver)
        assert X.shape == (1, 11)

    def test_zero_bilirubin_becomes_nan(self):
        data = HepatitisBInput(
            age=40, sex=0, bilirubin=0, alk_phosphate=90, sgot=30,
            albumin=4.0, protime=12, fatigue=0, malaise=0, ascites=0, varices=0,
        )
        X = preprocess_hepatitis_b_input(data)
        idx = HEPATITIS_B_FEATURE_ORDER.index("bilirubin")
        assert np.isnan(X[0, idx]), "Zero bilirubin should become NaN"

    def test_values_preserved(self, high_risk_liver):
        X = preprocess_hepatitis_b_input(high_risk_liver)
        assert X[0, HEPATITIS_B_FEATURE_ORDER.index("sgot")] == 180.0


class TestHepatitisBTriage:
    def test_high_risk_high_confidence_is_high(self):
        flag, _ = compute_triage("hepatitis_b", 0.88)
        assert flag == TriageFlag.HIGH

    def test_low_confidence_is_low(self):
        flag, _ = compute_triage("hepatitis_b", 0.45)
        assert flag == TriageFlag.LOW
