"""
api/routes/health.py
Health check endpoint - Django polls this to verify the ML service is up.

Endpoint: GET /api/ml/health
Returns:  status, which models are loaded, and device in use
"""
from fastapi import APIRouter
from api.schemas.prediction import HealthCheckResponse
from core.config import settings

router = APIRouter(prefix="/api/ml", tags=["Health"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="ML service health check",
)
async def health_check():
    # Import here to avoid circular imports at module load time
    try:
        from main import pneumonia_predictor
        pneumonia_loaded = pneumonia_predictor is not None
    except Exception:
        pneumonia_loaded = False

    return HealthCheckResponse(
        status="healthy" if pneumonia_loaded else "degraded",
        models_loaded={"pneumonia": pneumonia_loaded},
        device=settings.device,
    )
