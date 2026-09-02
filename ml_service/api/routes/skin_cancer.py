"""
api/routes/skin_cancer.py
FastAPI route for skin lesion (skin cancer) prediction.

Endpoint: POST /api/ml/predict/skin-cancer
Accepts:  multipart/form-data with a skin lesion image upload
Returns:  PredictionResponse JSON

Input is a dermoscopic or close-up smartphone photo of a skin lesion.
The model classifies it into one of 7 HAM10000 categories.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image
import io

from api.schemas.prediction import PredictionResponse
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Prediction"])


def get_skin_cancer_predictor():
    import ml_service.main as main_app
    return main_app.skin_cancer_predictor


@router.post(
    "/predict/skin-cancer",
    response_model=PredictionResponse,
    summary="Skin lesion (skin cancer) prediction",
    description=(
        "Upload a skin lesion photo (JPEG or PNG). Returns one of 7 lesion "
        "classes (e.g. Melanoma, Melanocytic Nevus, Basal Cell Carcinoma) "
        "with confidence score, differentials, triage flag, and a Grad-CAM "
        "heatmap highlighting the lesion region the model focused on."
    ),
)
async def predict_skin_cancer(
    file: UploadFile = File(..., description="Skin lesion image (JPEG/PNG, max 10MB)"),
    predictor=Depends(get_skin_cancer_predictor),
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
        logger.info(f"Running skin cancer inference on {file.filename}")
        result = predictor.predict(image)
        logger.info(
            f"Skin cancer prediction: {result.predicted_class} | "
            f"confidence={result.confidence_pct} | triage={result.triage}"
        )
        return result
    except Exception as e:
        logger.error(f"Skin cancer inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
