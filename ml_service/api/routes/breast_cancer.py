# api/routes/breast_cancer.py
# FastAPI route for Breast Cancer prediction.

# Endpoint: POST /api/ml/predict/breast-cancer
# Accepts:  multipart/form-data with a histology image upload
# Returns:  PredictionResponse JSON

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image
import io

from api.schemas.prediction import PredictionResponse
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Prediction"])


def get_breast_cancer_predictor():
    import ml_service.main as main_app

    return main_app.breast_cancer_predictor


@router.post(
    "/predict/breast-cancer",
    response_model=PredictionResponse,
    summary="Breast Cancer prediction",
    description=(
        "Upload a breast tissue histology slide image (JPEG or PNG). "
        "Returns BENIGN or MALIGNANT with confidence score, triage flag, "
        "and a Grad-CAM heatmap highlighting the cellular region the "
        "model focused on."
    ),
)
async def predict_breast_cancer(
    file: UploadFile = File(
        ..., description="Histology slide image (JPEG/PNG, max 10MB)"
    ),
    predictor=Depends(get_breast_cancer_predictor),
):
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type '{file.content_type}'. Only JPEG and PNG accepted.",
        )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail="File too large. Maximum allowed size is 10MB."
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read image: {str(e)}")

    try:
        logger.info(f"Running breast cancer inference on {file.filename}")
        result = predictor.predict(image)
        logger.info(
            f"Breast cancer prediction: {result.predicted_class} | "
            f"confidence={result.confidence_pct} | triage={result.triage}"
        )
        return result
    except Exception as e:
        logger.error(f"Breast cancer inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
