
# api/routes/anaemia.py
# FastAPI route for Anaemia prediction.

# Endpoint: POST /api/ml/predict/anaemia
# Accepts:  JSON body (AnaemiaInput)
# Returns:  PredictionResponse JSON

from fastapi import APIRouter, HTTPException, Depends
from preprocessing.tabular_transforms import AnaemiaInput
from api.schemas.prediction import PredictionResponse
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Prediction"])


def get_anaemia_predictor():
    import ml_service.main as main_app
    return main_app.anaemia_predictor


@router.post(
    "/predict/anaemia",
    response_model=PredictionResponse,
    summary="Anaemia prediction",
    description=(
        "Submit CBC-derived patient data (gender, haemoglobin, MCH, MCHC, MCV) "
        "as JSON. Returns ANAEMIC or NOT ANAEMIC with confidence score, "
        "triage flag, and SHAP feature importance."
    ),
)
async def predict_anaemia(
    data: AnaemiaInput,
    predictor=Depends(get_anaemia_predictor),
):
    try:
        logger.info(f"Running anaemia inference | hemoglobin={data.hemoglobin}")
        result = predictor.predict(data)
        logger.info(
            f"Anaemia prediction: {result.predicted_class} | "
            f"confidence={result.confidence_pct} | triage={result.triage}"
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Anaemia inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
