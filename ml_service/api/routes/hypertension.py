
# api/routes/hypertension.py
# FastAPI route for Hypertension prediction.

# Endpoint: POST /api/ml/predict/hypertension
# Accepts:  JSON body (HypertensionInput)
# Returns:  PredictionResponse JSON

from fastapi import APIRouter, HTTPException, Depends
from preprocessing.tabular_transforms import HypertensionInput
from api.schemas.prediction import PredictionResponse
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Prediction"])


def get_hypertension_predictor():
    import ml_service.main as main_app
    return main_app.hypertension_predictor


@router.post(
    "/predict/hypertension",
    response_model=PredictionResponse,
    summary="Hypertension prediction",
    description=(
        "Submit lifestyle and basic clinical risk data (age, salt intake, BMI, "
        "stress, sleep, smoking, family history, physical activity) as JSON. "
        "Returns HYPERTENSION or NO HYPERTENSION with confidence score, "
        "triage flag, and SHAP feature importance."
    ),
)
async def predict_hypertension(
    data: HypertensionInput,
    predictor=Depends(get_hypertension_predictor),
):
    try:
        logger.info(
            f"Running hypertension inference | bmi={data.bmi} age={data.age}")
        result = predictor.predict(data)
        logger.info(
            f"Hypertension prediction: {result.predicted_class} | "
            f"confidence={result.confidence_pct} | triage={result.triage}"
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Hypertension inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
