"""
api/routes/unified_image.py
Unified image-based diagnostic endpoint.

POST /api/ml/diagnose/image-based

Accepts:
  - file        : medical image (JPEG/PNG)
  - image_type  : "xray" | "skin" | "blood_smear" | "histology"
  - symptoms    : optional JSON string of SymptomsInput fields
  - clinical_notes : optional free text from the doctor

Returns:
  UnifiedDiagnosticResponse with all model results + Assistant summary
"""
import io
import json
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
from typing import Optional

from api.schemas.unified import (
    SymptomsInput, ImageType, UnifiedDiagnosticResponse, ModelResult
)
from api.schemas.prediction import TriageLevel
from core.router import route
from core.orchestrator import run_models
from core.gemini import generate_clinical_summary
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Unified Diagnosis"])

_TRIAGE_ORDER = {TriageLevel.LOW: 0, TriageLevel.MEDIUM: 1, TriageLevel.HIGH: 2}


@router.post(
    "/diagnose/image-based",
    response_model=UnifiedDiagnosticResponse,
    summary="Unified image-based diagnosis",
    description=(
        "Upload a medical image with optional structured symptoms and clinical notes. "
        "The system automatically routes to the appropriate model(s) based on image_type, "
        "runs any applicable tabular models in parallel, and returns a ranked result list "
        "with a Gemini-generated clinical summary."
    ),
)
async def diagnose_image_based(
    file: UploadFile = File(..., description="Medical image (JPEG/PNG, max 10MB)"),
    image_type: ImageType = Form(..., description="Type of image: xray | skin | blood_smear | histology"),
    symptoms: Optional[str] = Form(
        None,
        description="Optional JSON string of SymptomsInput fields (e.g. age, glucose, bmi)"
    ),
    clinical_notes: Optional[str] = Form(
        None,
        description="Doctor's free-text observations (e.g. 'patient reports itching for 3 weeks')"
    ),
):
    # Validate image 
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type '{file.content_type}'. Only JPEG and PNG accepted."
        )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum 10MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read image: {e}")

    # Parse optional symptoms JSON 
    symptoms_obj: Optional[SymptomsInput] = None
    if symptoms:
        try:
            symptoms_obj = SymptomsInput(**json.loads(symptoms))
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid symptoms JSON: {e}"
            )

    # Route 
    decision = route(symptoms_obj, image_type)
    logger.info(
        f"Image-based request | image_type={image_type.value} | "
        f"image_models={decision.image_models} | "
        f"tabular_models={decision.tabular_models} | "
        f"skipped={list(decision.skipped.keys())}"
    )

    #  Run models 
    results = await run_models(decision, image, symptoms_obj)

    if not results:
        raise HTTPException(
            status_code=422,
            detail="No models could run with the provided inputs. "
                   "Check that the image_type matches the uploaded image."
        )

    top_result = results[0]
    overall_triage = max(results, key=lambda r: _TRIAGE_ORDER[r.triage]).triage

    #  AI summary 
    results_for_gemini = [
        {
            "disease":        r.disease,
            "domain":         r.domain,
            "predicted_class": r.predicted_class,
            "confidence_pct": r.confidence_pct,
            "triage":         r.triage,
        }
        for r in results
    ]
    clinical_summary, gemini_used = await generate_clinical_summary(
        ml_results=results_for_gemini,
        clinical_notes=clinical_notes,
        image_type=image_type.value,
    )

    logger.info(
        f"Image diagnosis complete | top={top_result.disease} "
        f"{top_result.confidence_pct} | triage={overall_triage} | "
        f"gemini={'yes' if gemini_used else 'no'}"
    )

    return UnifiedDiagnosticResponse(
        request_id=uuid.uuid4().hex,
        models_run=decision.image_models + decision.tabular_models,
        models_skipped=[f"{k}: {v}" for k, v in decision.skipped.items()],
        results=results,
        top_result=top_result,
        overall_triage=overall_triage,
        clinical_summary=clinical_summary,
        gemini_used=gemini_used,
        image_type_used=image_type.value,
        clinical_notes=clinical_notes,
    )
