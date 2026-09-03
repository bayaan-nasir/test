from django.utils import timezone

from celery import shared_task

from .models import Inference, InferenceStatus
from .services.ml_client import MLClient, MLServiceError


@shared_task
def run_symptoms_inference(inference_id: int):
    try:
        inference = Inference.objects.get(pk=inference_id)
    except Inference.DoesNotExist:
        return

    _mark_processing(inference)

    try:
        result = MLClient().diagnose_symptoms(
            symptoms=inference.request_payload.get("symptoms", {}),
            clinical_notes=inference.request_payload.get("clinical_notes", ""),
        )

        _save_result(inference, result)

    except MLServiceError as exc:
        _mark_failed(inference, str(exc))

    except Exception as exc:
        _mark_failed(
            inference,
            f"Unexpected inference error: {exc}",
        )


@shared_task
def run_image_inference(inference_id: int):
    try:
        inference = Inference.objects.get(pk=inference_id)
    except Inference.DoesNotExist:
        return

    _mark_processing(inference)

    image = inference.images.first()

    if image is None:
        _mark_failed(
            inference,
            "No image was found for this inference.",
        )
        return

    try:
        with image.file.open("rb") as file:
            result = MLClient().diagnose_image(
                file=file,
                image_type=image.image_type,
                symptoms=inference.request_payload.get("symptoms"),
                clinical_notes=image.clinical_notes,
            )

        _save_result(inference, result)

    except MLServiceError as exc:
        _mark_failed(inference, str(exc))

    except Exception as exc:
        _mark_failed(
            inference,
            f"Unexpected inference error: {exc}",
        )


def _mark_processing(inference: Inference):
    inference.status = InferenceStatus.PROCESSING
    inference.started_at = timezone.now()
    inference.error_message = ""

    inference.save(
        update_fields=[
            "status",
            "started_at",
            "error_message",
        ]
    )


def _save_result(inference: Inference, result: dict):
    top_result = result.get("top_result") or {}

    inference.response_payload = result
    inference.fastapi_request_id = result.get("request_id")
    inference.overall_triage = result.get("overall_triage", "")
    inference.predicted_class = top_result.get("predicted_class", "")
    inference.confidence = top_result.get("confidence")
    inference.clinical_summary = result.get(
        "clinical_summary",
        "",
    )
    inference.status = InferenceStatus.COMPLETED
    inference.completed_at = timezone.now()
    inference.error_message = ""

    inference.save(
        update_fields=[
            "response_payload",
            "fastapi_request_id",
            "overall_triage",
            "predicted_class",
            "confidence",
            "clinical_summary",
            "status",
            "completed_at",
            "error_message",
        ]
    )


def _mark_failed(inference: Inference, message: str):
    inference.status = InferenceStatus.FAILED
    inference.error_message = message
    inference.completed_at = timezone.now()

    inference.save(
        update_fields=[
            "status",
            "error_message",
            "completed_at",
        ]
    )
