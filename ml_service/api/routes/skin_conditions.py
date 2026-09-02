"""
api/routes/skin_conditions.py
FastAPI route for general skin conditions prediction
(Eczema, Ringworm, Psoriasis, Acne, Normal Skin).

Endpoint: POST /api/ml/predict/skin-conditions
Accepts:  multipart/form-data with a skin photo upload
Returns:  PredictionResponse JSON
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image
import io

from api.schemas.prediction import PredictionResponse
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Prediction"])


def get_skin_conditions_predictor():
    import ml_service.main as main_app
    return main_app.skin_conditions_predictor


@router.post(
    "/predict/skin-conditions",
    response_model=PredictionResponse,
    summary="Skin conditions prediction (Eczema, Ringworm, etc.)",
    description=(
        "Upload a photo of an affected skin area (JPEG or PNG). Returns one "
        "of 5 classes (Eczema, Ringworm, Psoriasis, Acne, Normal Skin) with "
        "confidence score, differentials, triage flag, and a Grad-CAM heatmap."
    ),
)
async def predict_skin_conditions(
    file: UploadFile = File(..., description="Skin photo (JPEG/PNG, max 10MB)"),
    predictor=Depends(get_skin_conditions_predictor),
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
        logger.info(f"Running skin conditions inference on {file.filename}")
        result = predictor.predict(image)
        logger.info(
            f"Skin conditions prediction: {result.predicted_class} | "
            f"confidence={result.confidence_pct} | triage={result.triage}"
        )
        return result
    except Exception as e:
        logger.error(f"Skin conditions inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
