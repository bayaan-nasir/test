"""
main.py
FastAPI application entry point for the Ghana Health AI ML service.

PRIMARY ENDPOINTS (unified — use these):
  POST /api/ml/diagnose/image-based      image + optional symptoms + clinical notes
  POST /api/ml/diagnose/symptoms-based   symptoms JSON + clinical notes

LEGACY ENDPOINTS (individual — kept for debugging):
  POST /api/ml/predict/image             pneumonia only
  POST /api/ml/predict/malaria           malaria only
  POST /api/ml/predict/diabetes          diabetes only
  ... (all original per-disease routes)

Run (Windows CMD):
  uvicorn ml_service.main:app --host 0.0.0.0 --port 8001 --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path

from core.config import settings
from core.logger import logger

# Route imports 
from api.routes import (
    predict, health, malaria, diabetes,
    tuberculosis, heart_disease, skin_cancer, breast_cancer,
    covid19, anaemia, hypertension, hepatitis_b, skin_conditions,
)
from api.routes import unified_image, unified_symptoms

# Global predictor singletons 
# These are module-level dicts so route dependency functions
# always read the SAME reference — fixes the NoneType bug where
# from main import pneumonia_predictor imported None at module load.
_predictors: dict = {}

# Convenience accessors for existing route files that import by name
def _get(name):
    return _predictors.get(name)

@property
def pneumonia_predictor():      return _predictors.get("pneumonia")
@property
def malaria_predictor():        return _predictors.get("malaria")
@property
def diabetes_predictor():       return _predictors.get("diabetes")
@property
def tuberculosis_predictor():   return _predictors.get("tuberculosis")
@property
def heart_disease_predictor():  return _predictors.get("heart_disease")
@property
def skin_cancer_predictor():    return _predictors.get("skin_cancer")
@property
def breast_cancer_predictor():  return _predictors.get("breast_cancer")
@property
def covid19_predictor():        return _predictors.get("covid19")
@property
def anaemia_predictor():        return _predictors.get("anaemia")
@property
def hypertension_predictor():   return _predictors.get("hypertension")
@property
def hepatitis_b_predictor():    return _predictors.get("hepatitis_b")
@property
def skin_conditions_predictor(): return _predictors.get("skin_conditions")


# Also expose as plain module-level names for legacy route files 
# These are re-assigned after loading in lifespan below
pneumonia_predictor      = None
malaria_predictor        = None
diabetes_predictor       = None
tuberculosis_predictor   = None
heart_disease_predictor  = None
skin_cancer_predictor    = None
breast_cancer_predictor  = None
covid19_predictor        = None
anaemia_predictor        = None
hypertension_predictor   = None
hepatitis_b_predictor    = None
skin_conditions_predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pneumonia_predictor, malaria_predictor, diabetes_predictor
    global tuberculosis_predictor, heart_disease_predictor
    global skin_cancer_predictor, breast_cancer_predictor
    global covid19_predictor, anaemia_predictor, hypertension_predictor
    global hepatitis_b_predictor, skin_conditions_predictor

    logger.info("Starting Ghana Health AI ML service...")

    # Phase 1
    try:
        from models.pneumonia.inference import PneumoniaPredictor
        pneumonia_predictor = PneumoniaPredictor()
        _predictors["pneumonia"] = pneumonia_predictor
        logger.info(" Pneumonia model loaded")
    except Exception as e:
        logger.error(f" Pneumonia: {e}")

    try:
        from models.malaria.inference import MalariaPredictor
        malaria_predictor = MalariaPredictor()
        _predictors["malaria"] = malaria_predictor
        logger.info(" Malaria model loaded")
    except Exception as e:
        logger.error(f" Malaria: {e}")

    try:
        from models.diabetes.inference import DiabetesPredictor
        diabetes_predictor = DiabetesPredictor()
        _predictors["diabetes"] = diabetes_predictor
        logger.info(" Diabetes model loaded")
    except Exception as e:
        logger.error(f" Diabetes: {e}")

    # Phase 2
    try:
        from models.tuberculosis.inference import TuberculosisPredictor
        tuberculosis_predictor = TuberculosisPredictor()
        _predictors["tuberculosis"] = tuberculosis_predictor
        logger.info(" Tuberculosis model loaded")
    except Exception as e:
        logger.error(f" Tuberculosis: {e}")

    try:
        from models.heart_disease.inference import HeartDiseasePredictor
        heart_disease_predictor = HeartDiseasePredictor()
        _predictors["heart_disease"] = heart_disease_predictor
        logger.info(" Heart Disease model loaded")
    except Exception as e:
        logger.error(f" Heart Disease: {e}")

    try:
        from models.skin_cancer.inference import SkinCancerPredictor
        skin_cancer_predictor = SkinCancerPredictor()
        _predictors["skin_cancer"] = skin_cancer_predictor
        logger.info(" Skin Cancer model loaded")
    except Exception as e:
        logger.error(f" Skin Cancer: {e}")

    try:
        from models.breast_cancer.inference import BreastCancerPredictor
        breast_cancer_predictor = BreastCancerPredictor()
        _predictors["breast_cancer"] = breast_cancer_predictor
        logger.info(" Breast Cancer model loaded")
    except Exception as e:
        logger.error(f" Breast Cancer: {e}")

    # Phase 3
    try:
        from models.covid19.inference import Covid19Predictor
        covid19_predictor = Covid19Predictor()
        _predictors["covid19"] = covid19_predictor
        logger.info(" COVID-19 model loaded")
    except Exception as e:
        logger.error(f" COVID-19: {e}")

    try:
        from models.anaemia.inference import AnaemiaPredictor
        anaemia_predictor = AnaemiaPredictor()
        _predictors["anaemia"] = anaemia_predictor
        logger.info(" Anaemia model loaded")
    except Exception as e:
        logger.error(f" Anaemia: {e}")

    try:
        from models.hypertension.inference import HypertensionPredictor
        hypertension_predictor = HypertensionPredictor()
        _predictors["hypertension"] = hypertension_predictor
        logger.info(" Hypertension model loaded")
    except Exception as e:
        logger.error(f" Hypertension: {e}")

    try:
        from models.hepatitis_b.inference import HepatitisBPredictor
        hepatitis_b_predictor = HepatitisBPredictor()
        _predictors["hepatitis_b"] = hepatitis_b_predictor
        logger.info(" Hepatitis B model loaded")
    except Exception as e:
        logger.error(f" Hepatitis B: {e}")

    try:
        from models.skin_conditions.inference import SkinConditionsPredictor
        skin_conditions_predictor = SkinConditionsPredictor()
        _predictors["skin_conditions"] = skin_conditions_predictor
        logger.info(" Skin Conditions model loaded")
    except Exception as e:
        logger.error(f" Skin Conditions: {e}")

    loaded = len(_predictors)
    logger.info(f"ML service ready — {loaded}/12 models loaded")
    yield
    logger.info("Shutting down ML service...")


# App
app = FastAPI(
    title="Ghana Health AI - ML Service",
    description=(
        "**PRIMARY:** Use `/api/ml/diagnose/image-based` or "
        "`/api/ml/diagnose/symptoms-based` for unified hybrid diagnosis.\n\n"
        "Legacy per-disease endpoints remain available for debugging."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

Path("media").mkdir(exist_ok=True)
app.mount("/media", StaticFiles(directory="media"), name="media")

# Unified endpoints (primary) 
app.include_router(unified_image.router)
app.include_router(unified_symptoms.router)

# Legacy individual endpoints (debug / backwards compat) 
app.include_router(predict.router)
app.include_router(malaria.router)
app.include_router(diabetes.router)
app.include_router(tuberculosis.router)
app.include_router(heart_disease.router)
app.include_router(skin_cancer.router)
app.include_router(breast_cancer.router)
app.include_router(covid19.router)
app.include_router(anaemia.router)
app.include_router(hypertension.router)
app.include_router(hepatitis_b.router)
app.include_router(skin_conditions.router)
app.include_router(health.router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Ghana Health AI - ML Service",
        "version": "2.0.0",
        "primary_endpoints": {
            "image_based":    "/api/ml/diagnose/image-based",
            "symptoms_based": "/api/ml/diagnose/symptoms-based",
        },
        "docs":    "/docs",
        "health":  "/api/ml/health",
        "models":  f"{len(_predictors)}/12 loaded",
    }
