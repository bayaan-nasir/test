"""
core/orchestrator.py
Runs the models selected by the router and collects unified results.
"""
from PIL import Image
from api.schemas.unified import SymptomsInput, ModelResult
from api.schemas.prediction import TriageLevel
from core.router import RoutingDecision
from core.logger import logger

_TRIAGE_ORDER = {TriageLevel.LOW: 0, TriageLevel.MEDIUM: 1, TriageLevel.HIGH: 2}


def _worst_triage(results: list) -> TriageLevel:
    if not results:
        return TriageLevel.LOW
    return max(results, key=lambda r: _TRIAGE_ORDER[r.triage]).triage


def _normalise(pred, disease_name: str, domain: str, model_name: str) -> ModelResult:
    return ModelResult(
        disease=disease_name,
        domain=domain,
        predicted_class=pred.predicted_class,
        confidence=pred.confidence,
        confidence_pct=pred.confidence_pct,
        triage=pred.triage,
        explainability=pred.explainability.model_dump(),
        model_used=model_name,
    )


def _p(name: str):
    """Get a predictor from the active ml_service.main _predictors dict."""
    import ml_service.main as main_app
    return main_app._predictors.get(name)


async def run_models(
    decision: RoutingDecision,
    image,
    symptoms,
) -> list:
    """Run all models in the routing decision, return results sorted by confidence."""

    results = []

    # Image models
    if image is not None:

        if "pneumonia" in decision.image_models and _p("pneumonia"):
            try:
                pred = _p("pneumonia").predict(image)
                results.append(_normalise(pred, "Pneumonia", "respiratory", "DenseNet-121"))
            except Exception as e:
                logger.error(f"Pneumonia inference error: {e}")

        if "tuberculosis" in decision.image_models and _p("tuberculosis"):
            try:
                pred = _p("tuberculosis").predict(image)
                results.append(_normalise(pred, "Tuberculosis", "respiratory", "DenseNet-121"))
            except Exception as e:
                logger.error(f"TB inference error: {e}")

        if "covid19" in decision.image_models and _p("covid19"):
            try:
                pred = _p("covid19").predict(image)
                results.append(_normalise(pred, "COVID-19", "respiratory", "DenseNet-121"))
            except Exception as e:
                logger.error(f"COVID-19 inference error: {e}")

        if "malaria" in decision.image_models and _p("malaria"):
            try:
                pred = _p("malaria").predict(image)
                results.append(_normalise(pred, "Malaria", "infectious_diseases", "EfficientNet-B0"))
            except Exception as e:
                logger.error(f"Malaria inference error: {e}")

        if "skin_cancer" in decision.image_models and _p("skin_cancer"):
            try:
                pred = _p("skin_cancer").predict(image)
                results.append(_normalise(pred, "Skin Cancer", "dermatology", "EfficientNet-B3"))
            except Exception as e:
                logger.error(f"Skin cancer inference error: {e}")

        if "skin_conditions" in decision.image_models and _p("skin_conditions"):
            try:
                pred = _p("skin_conditions").predict(image)
                results.append(_normalise(pred, "Skin Conditions", "dermatology", "EfficientNet-B3"))
            except Exception as e:
                logger.error(f"Skin conditions inference error: {e}")

        if "breast_cancer" in decision.image_models and _p("breast_cancer"):
            try:
                pred = _p("breast_cancer").predict(image)
                results.append(_normalise(pred, "Breast Cancer", "oncology", "ResNet-50"))
            except Exception as e:
                logger.error(f"Breast cancer inference error: {e}")

    # Tabular models 
    if symptoms is not None:

        if "heart_disease" in decision.tabular_models and _p("heart_disease"):
            try:
                from preprocessing.tabular_transforms import HeartDiseaseInput
                data = HeartDiseaseInput(
                    age=symptoms.age or 50,
                    sex=symptoms.sex if symptoms.sex is not None else 1,
                    cp=symptoms.cp or 0,
                    trestbps=symptoms.trestbps or 120,
                    chol=symptoms.chol or 200,
                    fbs=symptoms.fbs or 0,
                    restecg=symptoms.restecg or 0,
                    thalach=symptoms.thalach or 150,
                    exang=symptoms.exang or 0,
                    oldpeak=symptoms.oldpeak or 0,
                    slope=symptoms.slope or 1,
                    ca=symptoms.ca or 0,
                    thal=symptoms.thal or 1,
                )
                pred = _p("heart_disease").predict(data)
                results.append(_normalise(pred, "Heart Disease", "cardiovascular", "XGBoost+RF"))
            except Exception as e:
                logger.error(f"Heart disease inference error: {e}")

        if "hypertension" in decision.tabular_models and _p("hypertension"):
            try:
                from preprocessing.tabular_transforms import HypertensionInput
                data = HypertensionInput(
                    age=symptoms.age or 40,
                    salt_intake=symptoms.salt_intake or 6,
                    bmi=symptoms.bmi or 25,
                    stress_score=symptoms.stress_score or 5,
                    sleep_hours=symptoms.sleep_hours or 7,
                    smoking=symptoms.smoking or 0,
                    family_history=symptoms.family_history or 0,
                    physical_activity=symptoms.physical_activity or 3,
                )
                pred = _p("hypertension").predict(data)
                results.append(_normalise(pred, "Hypertension", "cardiovascular", "XGBoost+RF"))
            except Exception as e:
                logger.error(f"Hypertension inference error: {e}")

        if "diabetes" in decision.tabular_models and _p("diabetes"):
            try:
                from preprocessing.tabular_transforms import DiabetesInput
                data = DiabetesInput(
                    pregnancies=symptoms.pregnancies or 0,
                    glucose=symptoms.glucose or 100,
                    blood_pressure=symptoms.trestbps or 70,
                    skin_thickness=symptoms.skin_thickness or 20,
                    insulin=symptoms.insulin or 80,
                    bmi=symptoms.bmi or 25,
                    diabetes_pedigree_function=symptoms.diabetes_pedigree_function or 0.3,
                    age=symptoms.age or 40,
                )
                pred = _p("diabetes").predict(data)
                results.append(_normalise(pred, "Diabetes", "metabolic", "XGBoost+RF"))
            except Exception as e:
                logger.error(f"Diabetes inference error: {e}")

        if "anaemia" in decision.tabular_models and _p("anaemia"):
            try:
                from preprocessing.tabular_transforms import AnaemiaInput
                data = AnaemiaInput(
                    gender=symptoms.sex if symptoms.sex is not None else 1,
                    hemoglobin=symptoms.hemoglobin or 13,
                    mch=symptoms.mch or 27,
                    mchc=symptoms.mchc or 32,
                    mcv=symptoms.mcv or 85,
                )
                pred = _p("anaemia").predict(data)
                results.append(_normalise(pred, "Anaemia", "metabolic", "XGBoost+RF"))
            except Exception as e:
                logger.error(f"Anaemia inference error: {e}")

        if "hepatitis_b" in decision.tabular_models and _p("hepatitis_b"):
            try:
                from preprocessing.tabular_transforms import HepatitisBInput
                data = HepatitisBInput(
                    age=symptoms.age or 40,
                    sex=symptoms.sex if symptoms.sex is not None else 1,
                    bilirubin=symptoms.bilirubin or 1.0,
                    alk_phosphate=symptoms.alk_phosphate or 80,
                    sgot=symptoms.sgot or 30,
                    albumin=symptoms.albumin or 4.0,
                    protime=symptoms.protime or 12,
                    fatigue=symptoms.fatigue or 0,
                    malaise=symptoms.malaise or 0,
                    ascites=symptoms.ascites or 0,
                    varices=symptoms.varices or 0,
                )
                pred = _p("hepatitis_b").predict(data)
                results.append(_normalise(pred, "Hepatitis B", "infectious_diseases", "XGBoost+RF"))
            except Exception as e:
                logger.error(f"Hepatitis B inference error: {e}")

    # Sort by confidence descending
    results.sort(key=lambda r: r.confidence, reverse=True)
    return results
