import json

from rest_framework import serializers

from accounts.models import UserRole
from patients.models import Patient

from .models import Inference, InferenceImage


class JSONDictField(serializers.Field):
    def to_internal_value(self, data):
        if data is None:
            return None

        if isinstance(data, dict):
            return data

        if isinstance(data, str):
            if not data.strip():
                raise serializers.ValidationError("Must be a valid JSON object.")
            try:
                parsed = json.loads(data)
            except (TypeError, ValueError) as exc:
                raise serializers.ValidationError(
                    "Must be a valid JSON object."
                ) from exc
            if not isinstance(parsed, dict):
                raise serializers.ValidationError("Must decode to a JSON object.")
            return parsed

        raise serializers.ValidationError("Must be a JSON object.")


class InferenceImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = InferenceImage
        fields = (
            "id",
            "file",
            "image_type",
            "clinical_notes",
            "created_at",
        )
        read_only_fields = (
            "id",
            "created_at",
        )


class InferenceSerializer(serializers.ModelSerializer):
    patient_id = serializers.SerializerMethodField()
    patient_name = serializers.SerializerMethodField()

    class Meta:
        model = Inference
        fields = (
            "id",
            "patient_id",
            "patient_name",
            "inference_type",
            "status",
            "overall_triage",
            "predicted_class",
            "confidence",
            "clinical_summary",
            "request_payload",
            "response_payload",
            "error_message",
            "fastapi_request_id",
            "created_at",
            "started_at",
            "completed_at",
        )
        read_only_fields = fields

    def get_patient_id(self, obj):
        return obj.patient.patient_id if obj.patient else None

    def get_patient_name(self, obj):
        if not obj.patient:
            return "Anonymous patient"
        return f"{obj.patient.first_name} {obj.patient.last_name}"


class CreateSymptomsInferenceSerializer(serializers.Serializer):
    patient_id = serializers.CharField(
        required=False,
        allow_null=True,
        allow_blank=True,
    )
    clinical_notes = serializers.CharField(
        required=False,
        allow_blank=True,
        default="",
    )
    symptoms = JSONDictField(
        required=False,
        default=None,
        allow_null=True,
    )
    models = JSONDictField(
        required=False,
        default=None,
        allow_null=True,
    )

    def _get_patient(self, value):
        try:
            return Patient.objects.get(patient_id=value)
        except Patient.DoesNotExist as exc:
            raise serializers.ValidationError("Patient not found.") from exc

    def validate_patient_id(self, value):
        if not value:
            return None

        patient = self._get_patient(value)
        request = self.context.get("request")

        if request is None or not getattr(request, "user", None):
            return value

        if request.user.role == UserRole.ADMIN:
            return value

        has_access = (
            patient.created_by_id == request.user.id
            or patient.assignments.filter(clinician=request.user).exists()
        )

        if not has_access:
            raise serializers.ValidationError("You do not have access to this patient.")

        return value

    def _normalize_payload(self, value, field_name):
        if value is None:
            return None

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise serializers.ValidationError(
                    {field_name: "Must be a valid JSON object."}
                ) from exc
            if not isinstance(parsed, dict):
                raise serializers.ValidationError(
                    {field_name: "Must decode to a JSON object."}
                )
            return parsed

        if isinstance(value, dict):
            return value

        raise serializers.ValidationError({field_name: "Must be a JSON object."})

    def validate(self, attrs):
        model_payload = self._normalize_payload(attrs.get("symptoms"), "symptoms")
        legacy_payload = self._normalize_payload(attrs.get("models"), "models")

        if model_payload is None and legacy_payload is None:
            raise serializers.ValidationError(
                {"symptoms": "At least one symptom/model payload is required."}
            )

        payload = legacy_payload if model_payload is None else model_payload

        if payload is None or not isinstance(payload, dict):
            raise serializers.ValidationError(
                {"symptoms": "Symptoms input must be an object."}
            )

        if not payload:
            raise serializers.ValidationError(
                {"symptoms": "Symptoms input cannot be empty."}
            )

        attrs["symptoms"] = payload
        attrs.pop("models", None)
        return attrs


class CreateImageInferenceSerializer(serializers.Serializer):
    patient_id = serializers.CharField(
        required=False,
        allow_null=True,
        allow_blank=True,
    )
    file = serializers.FileField()
    image_type = serializers.CharField()
    clinical_notes = serializers.CharField(
        required=False,
        allow_blank=True,
        default="",
    )
    symptoms = JSONDictField(
        required=False,
        default=dict,
        allow_null=True,
    )

    def _get_patient(self, value):
        try:
            return Patient.objects.get(patient_id=value)
        except Patient.DoesNotExist as exc:
            raise serializers.ValidationError("Patient not found.") from exc

    def validate_patient_id(self, value):
        if not value:
            return None

        patient = self._get_patient(value)
        request = self.context.get("request")

        if request is None or not getattr(request, "user", None):
            return value

        if request.user.role == UserRole.ADMIN:
            return value

        has_access = (
            patient.created_by_id == request.user.id
            or patient.assignments.filter(clinician=request.user).exists()
        )

        if not has_access:
            raise serializers.ValidationError("You do not have access to this patient.")

        return value

    def validate(self, attrs):
        symptoms = attrs.get("symptoms")
        normalized = (
            self._normalize_payload(symptoms, "symptoms")
            if symptoms is not None
            else {}
        )

        if normalized and not isinstance(normalized, dict):
            raise serializers.ValidationError(
                {"symptoms": "Symptoms input must be an object."}
            )

        attrs["symptoms"] = normalized
        return attrs

    def _normalize_payload(self, value, field_name):
        if value is None:
            return None

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise serializers.ValidationError(
                    {field_name: "Must be a valid JSON object."}
                ) from exc
            if not isinstance(parsed, dict):
                raise serializers.ValidationError(
                    {field_name: "Must decode to a JSON object."}
                )
            return parsed

        if isinstance(value, dict):
            return value

        raise serializers.ValidationError({field_name: "Must be a JSON object."})
