"""
api/routes/unified_symptoms.py
Unified symptoms-based diagnostic endpoint.

POST /api/ml/diagnose/symptoms-based

Accepts:
  - symptoms      : JSON body of SymptomsInput fields
  - clinical_notes: optional free text from the doctor

The router decides which tabular models have enough data to run.
All image models are skipped (no image provided).
Returns UnifiedDiagnosticResponse with ranked results + Gemini summary.
"""

import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from api.schemas.unified import SymptomsInput, UnifiedDiagnosticResponse
from api.schemas.prediction import TriageLevel
from core.router import route
from core.orchestrator import run_models
from core.gemini import generate_clinical_summary
from core.logger import logger

router = APIRouter(prefix="/api/ml", tags=["Unified Diagnosis"])

_TRIAGE_ORDER = {TriageLevel.LOW: 0, TriageLevel.MEDIUM: 1, TriageLevel.HIGH: 2}


class SymptomsDiagnosticRequest(BaseModel):
    """
    Request body for the symptoms-only endpoint.
    symptoms is the full structured form; clinical_notes is free text.
    """

    symptoms: SymptomsInput
    clinical_notes: Optional[str] = None


@router.post(
    "/diagnose/symptoms-based",
    response_model=UnifiedDiagnosticResponse,
    summary="Unified symptoms-based diagnosis",
    description=(
        "Submit structured patient data (lab values, vitals, lifestyle factors) "
        "as JSON, with optional clinical notes. The system automatically determines "
        "which tabular models have sufficient data to run, executes them in parallel, "
        "ranks results by confidence, and returns a Gemini-generated clinical summary. "
        "All image-based models (chest X-ray, blood smear, histology, skin) are skipped "
        "since no image was provided."
    ),
)
async def diagnose_symptoms_based(request: SymptomsDiagnosticRequest):

    symptoms = request.symptoms
    clinical_notes = request.clinical_notes

    # Route to determine which models can run based on provided symptoms
    decision = route(symptoms, image_type=None)

    logger.info(
        f"Symptoms-based request | "
        f"tabular_models={decision.tabular_models} | "
        f"skipped={list(decision.skipped.keys())}"
    )

    if not decision.tabular_models:
        raise HTTPException(
            status_code=422,
            detail=(
                "Not enough clinical data to run any models. "
                "Please provide at least one of the following groups of fields:\n"
                "• Cardiovascular: age, cp, trestbps, chol, thalach (≥5 fields)\n"
                "• Metabolic/Diabetes: glucose + bmi (≥2 fields)\n"
                "• Anaemia (CBC): hemoglobin\n"
                "• Lifestyle/Hypertension: age, bmi, salt_intake (≥3 fields)\n"
                "• Liver/Hepatitis B: bilirubin, sgot, albumin (≥3 fields)"
            ),
        )

    # Run tabular models
    results = await run_models(decision, image=None, symptoms=symptoms)

    if not results:
        raise HTTPException(
            status_code=500,
            detail="All models failed during inference. Check server logs.",
        )

    top_result = results[0]
    overall_triage = max(results, key=lambda r: _TRIAGE_ORDER[r.triage]).triage

    #  AI summary
    results_for_gemini = [
        {
            "disease": r.disease,
            "domain": r.domain,
            "predicted_class": r.predicted_class,
            "confidence_pct": r.confidence_pct,
            "triage": r.triage,
        }
        for r in results
    ]
    clinical_summary, gemini_used = await generate_clinical_summary(
        ml_results=results_for_gemini,
        clinical_notes=clinical_notes,
        image_type=None,
    )

    logger.info(
        f"Symptoms diagnosis complete | models_run={decision.tabular_models} | "
        f"top={top_result.disease} {top_result.confidence_pct} | "
        f"triage={overall_triage} | gemini={'yes' if gemini_used else 'no'}"
    )

    return UnifiedDiagnosticResponse(
        request_id=uuid.uuid4().hex,
        models_run=decision.tabular_models,
        models_skipped=[f"{k}: {v}" for k, v in decision.skipped.items()],
        results=results,
        top_result=top_result,
        overall_triage=overall_triage,
        clinical_summary=clinical_summary,
        gemini_used=gemini_used,
        image_type_used=None,
        clinical_notes=clinical_notes,
    )
