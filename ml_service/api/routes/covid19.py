
# api/routes/covid19.py
# FastAPI route for COVID-19 prediction.

# Endpoint: POST /api/ml/predict/covid19
# Accepts:  multipart/form-data with a chest X-ray image upload
# Returns:  PredictionResponse JSON (3-class: NORMAL / COVID-19 / PNEUMONIA)

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image
import io

from api.schemas.prediction import PredictionResponse
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Prediction"])


def get_covid19_predictor():
    import ml_service.main as main_app
    return main_app.covid19_predictor


@router.post(
    "/predict/covid19",
    response_model=PredictionResponse,
    summary="COVID-19 prediction",
    description=(
        "Upload a chest X-ray image (JPEG or PNG). Returns NORMAL, COVID-19, "
        "or PNEUMONIA with confidence score, differentials, triage flag, "
        "and a Grad-CAM heatmap."
    ),
)
async def predict_covid19(
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG, max 10MB)"),
    predictor=Depends(get_covid19_predictor),
):
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type '{file.content_type}'. Only JPEG and PNG accepted."
        )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 10MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read image: {str(e)}")

    try:
        logger.info(f"Running COVID-19 inference on {file.filename}")
        result = predictor.predict(image)
        logger.info(
            f"COVID-19 prediction: {result.predicted_class} | "
            f"confidence={result.confidence_pct} | triage={result.triage}"
        )
        return result
    except Exception as e:
        logger.error(f"COVID-19 inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
