
# api/routes/hepatitis_b.py
# FastAPI route for Hepatitis B prediction.

# Endpoint: POST /api/ml/predict/hepatitis-b
# Accepts:  JSON body (HepatitisBInput)
# Returns:  PredictionResponse JSON

from fastapi import APIRouter, HTTPException, Depends
from preprocessing.tabular_transforms import HepatitisBInput
from api.schemas.prediction import PredictionResponse
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Prediction"])


def get_hepatitis_b_predictor():
    import ml_service.main as main_app
    return main_app.hepatitis_b_predictor


@router.post(
    "/predict/hepatitis-b",
    response_model=PredictionResponse,
    summary="Hepatitis B prediction",
    description=(
        "Submit clinical and lab data (bilirubin, liver enzymes, albumin, "
        "prothrombin time, and symptoms) as JSON. Returns HIGH RISK or "
        "LOW RISK with confidence score, triage flag, and SHAP feature "
        "importance."
    ),
)
async def predict_hepatitis_b(
    data: HepatitisBInput,
    predictor=Depends(get_hepatitis_b_predictor),
):
    try:
        logger.info(f"Running hepatitis B inference | bilirubin={data.bilirubin} sgot={data.sgot}")
        result = predictor.predict(data)
        logger.info(
            f"Hepatitis B prediction: {result.predicted_class} | "
            f"confidence={result.confidence_pct} | triage={result.triage}"
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Hepatitis B inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
