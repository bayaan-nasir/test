from datetime import timezone
import json

import requests
from django.conf import settings

from inference.models import Inference, InferenceStatus


class MLServiceError(Exception):
    """Raised when communication with the ML service fails."""


class MLClient:
    def __init__(self):
        self.base_url = settings.ML_SERVICE_URL.rstrip("/")

    def diagnose_symptoms(
        self,
        *,
        symptoms: dict,
        clinical_notes: str = "",
        timeout: int = 120,
    ) -> dict:
        url = f"{self.base_url}/api/ml/diagnose/symptoms-based"

        payload = {
            "symptoms": symptoms,
            "clinical_notes": clinical_notes or None,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise MLServiceError(f"Unable to connect to ML service: {exc}") from exc

        return self._handle_response(response)

    def diagnose_image(
        self,
        *,
        file,
        image_type: str,
        symptoms: dict | None = None,
        clinical_notes: str = "",
        timeout: int = 180,
    ) -> dict:
        url = f"{self.base_url}/api/ml/diagnose/image-based"

        files = {
            "file": (
                file.name,
                file,
                getattr(file, "content_type", None),
            )
        }

        data = {
            "image_type": image_type,
            "clinical_notes": clinical_notes or "",
        }

        if symptoms:
            data["symptoms"] = json.dumps(symptoms)

        try:
            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise MLServiceError(f"Unable to connect to ML service: {exc}") from exc

        return self._handle_response(response)

    @staticmethod
    def _handle_response(response) -> dict:
        try:
            data = response.json()
        except ValueError as exc:
            raise MLServiceError(
                "ML service returned an invalid JSON response."
            ) from exc

        if not response.ok:
            detail = data.get(
                "detail",
                "ML service request failed.",
            )

            raise MLServiceError(str(detail))

        return data

    def _save_result(inference: Inference, result: dict):
        top_result = result.get("top_result") or {}

        inference.response_payload = result
        inference.fastapi_request_id = result.get("request_id")
        inference.overall_triage = result.get("overall_triage", "")
        inference.predicted_class = top_result.get("predicted_class", "")
        inference.confidence = top_result.get("confidence_pct")
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
