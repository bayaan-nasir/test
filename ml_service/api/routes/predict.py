"""
api/routes/predict.py
FastAPI route for image-based disease prediction.

Endpoint: POST /api/ml/predict/image
Accepts:  multipart/form-data with an image file upload
Returns:  PredictionResponse JSON

This is what Django calls when a user submits a chest X-ray for pneumonia prediction.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import io

from api.schemas.prediction import PredictionResponse
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Prediction"])


def get_predictor():
    """
    Dependency injection — returns the predictor loaded at startup.
    Imported lazily to avoid circular imports.
    """
    import ml_service.main as main_app
    return main_app.pneumonia_predictor


@router.post(
    "/predict/image",
    response_model=PredictionResponse,
    summary="Image-based disease prediction",
    description=(
        "Upload a chest X-ray image (JPEG or PNG). "
        "Returns predicted disease, confidence score, differential diagnoses, "
        "triage flag, and a Grad-CAM heatmap URL."
    ),
)
async def predict_image(
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG, max 10MB)"),
    predictor=Depends(get_predictor),
):
    #  Validate file type 
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type '{file.content_type}'. Only JPEG and PNG accepted."
        )

    #  Validate file size (max about 10 MB) 
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 10MB."
        )

    # Load image 
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read image: {str(e)}")

    #  Run inference 
    try:
        logger.info(f"Running pneumonia inference on uploaded image ({file.filename})")
        result = predictor.predict(image)
        logger.info(
            f"Prediction: {result.predicted_class} | "
            f"confidence={result.confidence_pct} | triage={result.triage}"
        )
        return result
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
