import json
import logging

from django.db import transaction
from django.db.models import Q
from rest_framework import generics, permissions, status
from rest_framework.views import APIView
from .services.ml_client import MLClient, MLServiceError
from rest_framework.response import Response

from accounts.models import UserRole
from patients.models import Patient

from .models import Inference, InferenceImage, InferenceType
from .serializers import (
    CreateImageInferenceSerializer,
    CreateSymptomsInferenceSerializer,
    InferenceSerializer,
)
from .tasks import run_image_inference, run_symptoms_inference
from .services.ml_client import MLClient, MLServiceError
from .analytics import compute_model_usage_stats

logger = logging.getLogger(__name__)


def _coerce_symptoms(value):
    if value is None:
        return {}

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}

        if isinstance(parsed, dict):
            return parsed

        return {}

    if isinstance(value, dict):
        return value

    return {}


def _enqueue_inference_task(task, inference_id):
    transaction.on_commit(lambda: task.delay(inference_id))


class InferenceListView(generics.ListAPIView):
    serializer_class = InferenceSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        queryset = Inference.objects.select_related("patient")

        if self.request.user.role == UserRole.ADMIN:
            return queryset

        return queryset.filter(
            Q(requested_by=self.request.user)
            | Q(patient__assignments__clinician=self.request.user)
        ).distinct()


class InferenceDetailView(generics.RetrieveAPIView):
    serializer_class = InferenceSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        queryset = Inference.objects.select_related("patient")

        if self.request.user.role == UserRole.ADMIN:
            return queryset

        return queryset.filter(
            Q(requested_by=self.request.user)
            | Q(patient__assignments__clinician=self.request.user)
        ).distinct()


class SymptomsInferenceCreateView(generics.CreateAPIView):
    serializer_class = CreateSymptomsInferenceSerializer
    permission_classes = [permissions.IsAuthenticated]

    @transaction.atomic
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(
            data=request.data,
            context={"request": request},
        )
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data

        patient = None
        patient_id = data.get("patient_id")

        if patient_id:
            patient = Patient.objects.get(patient_id=patient_id)

        payload = {
            "symptoms": _coerce_symptoms(data.get("symptoms")),
            "clinical_notes": data.get("clinical_notes", ""),
        }

        inference = Inference.objects.create(
            patient=patient,
            requested_by=request.user,
            inference_type=InferenceType.SYMPTOMS,
            request_payload=payload,
        )

        _enqueue_inference_task(
            run_symptoms_inference,
            inference.id,
        )

        return Response(
            InferenceSerializer(inference).data,
            status=status.HTTP_202_ACCEPTED,
        )


class ImageInferenceCreateView(generics.CreateAPIView):
    serializer_class = CreateImageInferenceSerializer
    permission_classes = [permissions.IsAuthenticated]

    @transaction.atomic
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(
            data=request.data,
            context={"request": request},
        )
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data

        patient = None
        patient_id = data.get("patient_id")

        if patient_id:
            patient = Patient.objects.get(patient_id=patient_id)

        inference = Inference.objects.create(
            patient=patient,
            requested_by=request.user,
            inference_type=InferenceType.IMAGE,
            request_payload={
                "symptoms": _coerce_symptoms(data.get("symptoms")),
                "clinical_notes": data.get("clinical_notes", ""),
                "image_type": data["image_type"],
            },
        )

        InferenceImage.objects.create(
            inference=inference,
            file=data["file"],
            image_type=data["image_type"],
            clinical_notes=data.get("clinical_notes", ""),
        )

        _enqueue_inference_task(
            run_image_inference,
            inference.id,
        )

        return Response(
            InferenceSerializer(inference).data,
            status=status.HTTP_202_ACCEPTED,
        )


class ModelRegistryView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            data = MLClient().get_model_registry()
        except MLServiceError as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response(data)


class ModelRegistryView(APIView):
    """Proxies the live ML service model registry so the frontend never
    has to hardcode which models exist or what fields they accept."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            data = MLClient().get_model_registry()
        except MLServiceError as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response(data)


class ModelStatsView(APIView):
    """Real per-model usage stats derived from completed Inference records
    — see analytics.py for exactly what's computed and its limitations."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        return Response(compute_model_usage_stats())
