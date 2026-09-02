from celery import shared_task
from django.utils import timezone

from .models import (
    Inference,
    InferenceImage,
    InferenceStatus,
)
from .services.ml_client import (
    MLClient,
    MLServiceError,
)


@shared_task
def run_symptoms_inference(
    inference_id: int,
):
    inference = Inference.objects.get(
        pk=inference_id,
    )

    inference.status = InferenceStatus.PROCESSING
    inference.started_at = timezone.now()
    inference.save(
        update_fields=[
            "status",
            "started_at",
        ]
    )

    try:
        client = MLClient()

        result = client.diagnose_symptoms(
            inference.request_payload,
        )

        _save_result(
            inference,
            result,
        )

    except MLServiceError as exc:
        _mark_failed(
            inference,
            str(exc),
        )

    except Exception as exc:
        _mark_failed(
            inference,
            f"Unexpected inference error: {exc}",
        )


@shared_task
def run_image_inference(
    inference_id: int,
):
    inference = Inference.objects.get(
        pk=inference_id,
    )

    inference.status = InferenceStatus.PROCESSING
    inference.started_at = timezone.now()
    inference.save(
        update_fields=[
            "status",
            "started_at",
        ]
    )

    try:
        image = inference.images.first()

        if image is None:
            raise MLServiceError("No image was attached to this inference.")

        with image.file.open("rb") as file:
            client = MLClient()

            result = client.diagnose_image(
                file=file,
                image_type=image.image_type,
                clinical_notes=image.clinical_notes,
            )

        _save_result(
            inference,
            result,
        )

    except MLServiceError as exc:
        _mark_failed(
            inference,
            str(exc),
        )

    except Exception as exc:
        _mark_failed(
            inference,
            f"Unexpected inference error: {exc}",
        )


def _save_result(
    inference: Inference,
    result: dict,
):
    top_result = result.get("top_result") or {}

    inference.response_payload = result

    inference.fastapi_request_id = result.get("request_id")

    inference.overall_triage = result.get("overall_triage") or ""

    inference.predicted_class = top_result.get("predicted_class") or ""

    inference.confidence = top_result.get("confidence")

    inference.clinical_summary = result.get("clinical_summary") or ""

    inference.status = InferenceStatus.COMPLETED

    inference.completed_at = timezone.now()

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
        ]
    )


def _mark_failed(
    inference: Inference,
    message: str,
):
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
