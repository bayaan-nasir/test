"""
inference/analytics.py
Derives real per-model usage statistics from stored Inference records.

No separate metrics pipeline exists (or is needed) — every completed
unified diagnosis run already stores response_payload["results"], which
includes disease, confidence, triage, and model_used per model that ran.
This aggregates that existing data rather than fabricating numbers.

Caveats, stated honestly rather than hidden:
  - avg_confidence_pct is the model's own softmax/predict_proba confidence,
    NOT validated accuracy. No ground-truth evaluation pipeline exists.
  - avg_latency_seconds is timed at the whole unified-request level
    (started_at → completed_at), not per individual model within a
    multi-model run. If 3 models ran in one request, all 3 get credited
    with that request's total duration.
"""

from collections import defaultdict
from statistics import mean

from .models import Inference, InferenceStatus

# Maps the display-name "disease" strings used in ml_service/core/orchestrator.py
# onto the model_id keys used in ml_service/api/routes/model_registry.py.
DISEASE_TO_MODEL_ID = {
    "Heart Disease": "heart_disease",
    "Hypertension": "hypertension",
    "Diabetes": "diabetes",
    "Anaemia": "anaemia",
    "Hepatitis B": "hepatitis_b",
    "Pneumonia": "pneumonia",
    "Tuberculosis": "tuberculosis",
    "COVID-19": "covid19",
    "Malaria": "malaria",
    "Skin Cancer": "skin_cancer",
    "Skin Conditions": "skin_conditions",
    "Breast Cancer": "breast_cancer",
}

RECENT_RUNS_LIMIT = 5


def compute_model_usage_stats() -> dict:
    confidences: dict[str, list[float]] = defaultdict(list)
    run_counts: dict[str, int] = defaultdict(int)
    latencies: dict[str, list[float]] = defaultdict(list)
    last_run: dict[str, str] = {}
    recent_runs: dict[str, list[dict]] = defaultdict(list)

    completed = (
        Inference.objects.filter(status=InferenceStatus.COMPLETED)
        .select_related("patient")
        .order_by("-created_at")
    )

    for inference in completed.iterator():
        results = (inference.response_payload or {}).get("results") or []

        duration = None
        if inference.started_at and inference.completed_at:
            duration = (inference.completed_at - inference.started_at).total_seconds()

        patient_name = (
            f"{inference.patient.first_name} {inference.patient.last_name}"
            if inference.patient
            else "Anonymous patient"
        )
        created = inference.created_at.isoformat() if inference.created_at else None

        for result in results:
            model_id = DISEASE_TO_MODEL_ID.get(result.get("disease"))
            if model_id is None:
                continue

            confidence = result.get("confidence")
            if isinstance(confidence, (int, float)):
                confidences[model_id].append(float(confidence))

            run_counts[model_id] += 1

            if duration is not None:
                latencies[model_id].append(duration)

            if created and (model_id not in last_run or created > last_run[model_id]):
                last_run[model_id] = created

            if len(recent_runs[model_id]) < RECENT_RUNS_LIMIT:
                recent_runs[model_id].append(
                    {
                        "inference_id": inference.id,
                        "patient_name": patient_name,
                        "predicted_class": result.get("predicted_class"),
                        "confidence_pct": result.get("confidence_pct"),
                        "triage": result.get("triage"),
                        "created_at": created,
                    }
                )

    stats = {}
    for model_id in set(DISEASE_TO_MODEL_ID.values()):
        stats[model_id] = {
            "total_runs": run_counts.get(model_id, 0),
            "avg_confidence_pct": (
                round(mean(confidences[model_id]) * 100, 1)
                if confidences.get(model_id)
                else None
            ),
            "avg_latency_seconds": (
                round(mean(latencies[model_id]), 2) if latencies.get(model_id) else None
            ),
            "last_run_at": last_run.get(model_id),
            "recent_runs": recent_runs.get(model_id, []),
        }

    return stats
