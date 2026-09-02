
# api/routes/heart_disease.py
# FastAPI route for Heart Disease prediction.

# Endpoint: POST /api/ml/predict/heart-disease
# Accepts:  JSON body (HeartDiseaseInput)
# Returns:  PredictionResponse JSON

from fastapi import APIRouter, HTTPException, Depends
from preprocessing.tabular_transforms import HeartDiseaseInput
from api.schemas.prediction import PredictionResponse
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Prediction"])


def get_heart_disease_predictor():
    import ml_service.main as main_app
    return main_app.heart_disease_predictor


@router.post(
    "/predict/heart-disease",
    response_model=PredictionResponse,
    summary="Heart Disease prediction",
    description=(
        "Submit structured cardiovascular patient data (age, blood pressure, "
        "cholesterol, ECG results, etc.) as JSON. Returns HEART DISEASE or "
        "NO HEART DISEASE with confidence score, triage flag, and SHAP "
        "feature importance."
    ),
)
async def predict_heart_disease(
    data: HeartDiseaseInput,
    predictor=Depends(get_heart_disease_predictor),
):
    try:
        logger.info(
            f"Running heart disease inference | "
            f"age={data.age} chol={data.chol} thalach={data.thalach}"
        )
        result = predictor.predict(data)
        logger.info(
            f"Heart disease prediction: {result.predicted_class} | "
            f"confidence={result.confidence_pct} | triage={result.triage}"
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Heart disease inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
