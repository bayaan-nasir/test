
# api/routes/diabetes.py
# FastAPI route for Type 2 Diabetes prediction.

# Endpoint: POST /api/ml/predict/diabetes
# Accepts:  JSON body (DiabetesInput)
# Returns:  PredictionResponse JSON

from fastapi import APIRouter, HTTPException, Depends
from preprocessing.tabular_transforms import DiabetesInput
from api.schemas.prediction import PredictionResponse
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Prediction"])


def get_diabetes_predictor():
    import ml_service.main as main_app
    return main_app.diabetes_predictor


@router.post(
    "/predict/diabetes",
    response_model=PredictionResponse,
    summary="Type 2 Diabetes prediction",
    description=(
        "Submit structured patient data (glucose, BMI, age, etc.) as JSON. "
        "Returns DIABETIC or NON-DIABETIC with confidence score, "
        "triage flag, and SHAP feature importance showing which clinical "
        "values influenced the prediction most."
    ),
)
async def predict_diabetes(
    data: DiabetesInput,
    predictor=Depends(get_diabetes_predictor),
):
    try:
        logger.info(
            f"Running diabetes inference | "
            f"glucose={data.glucose} bmi={data.bmi} age={data.age}"
        )
        result = predictor.predict(data)
        logger.info(
            f"Diabetes prediction: {result.predicted_class} | "
            f"confidence={result.confidence_pct} | triage={result.triage}"
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Diabetes inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
