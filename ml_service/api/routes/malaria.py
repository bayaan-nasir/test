"""
api/routes/malaria.py
FastAPI route for malaria blood smear prediction.

Endpoint: POST /api/ml/predict/malaria
Accepts:  multipart/form-data with a cell image upload
Returns:  PredictionResponse JSON

The input is a blood smear microscopy image - a single cell or a region
of a slide. The model classifies it as PARASITIZED or UNINFECTED.

Clinical context:
  In Ghana, microscopy of Giemsa-stained blood smears is the gold-standard
  diagnostic for malaria. This endpoint automates that reading step,
  returning a result with a Grad-CAM heatmap showing the parasite region.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image
import io

from api.schemas.prediction import PredictionResponse
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Prediction"])


def get_malaria_predictor():
    import ml_service.main as main_app
    return main_app.malaria_predictor


@router.post(
    "/predict/malaria",
    response_model=PredictionResponse,
    summary="Malaria blood smear prediction",
    description=(
        "Upload a blood smear microscopy cell image (JPEG or PNG). "
        "Returns PARASITIZED or UNINFECTED with confidence score, "
        "triage flag, and a Grad-CAM heatmap highlighting the parasite region."
    ),
)
async def predict_malaria(
    file: UploadFile = File(
        ...,
        description="Blood smear cell image (JPEG/PNG, max 5MB)"
    ),
    predictor=Depends(get_malaria_predictor),
):
    # To Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type '{file.content_type}'. Only JPEG and PNG accepted."
        )

    # Validate file size (max about 5MB - cell images are small)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 5MB."
        )

    # Load image
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read image: {str(e)}")

    # Run inference
    try:
        logger.info(f"Running malaria inference on {file.filename}")
        result = predictor.predict(image)
        logger.info(
            f"Malaria prediction: {result.predicted_class} | "
            f"confidence={result.confidence_pct} | triage={result.triage}"
        )
        return result
    except Exception as e:
        logger.error(f"Malaria inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
