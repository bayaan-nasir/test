from rest_framework import serializers

from accounts.models import UserRole

from .models import Patient, ClinicalRecord, PatientAssignment


class PatientSerializer(serializers.ModelSerializer):
    full_name = serializers.SerializerMethodField()

    class Meta:
        model = Patient
        fields = [
            "patient_id",
            "first_name",
            "last_name",
            "full_name",
            "date_of_birth",
            "sex",
            "phone_number",
            "email",
            "notes",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "patient_id",
            "full_name",
            "created_at",
            "updated_at",
        ]

    def get_full_name(self, obj):
        return f"{obj.first_name} {obj.last_name}".strip()


class PatientCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = [
            "first_name",
            "last_name",
            "date_of_birth",
            "sex",
            "phone_number",
            "email",
            "notes",
        ]


class ClinicalRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClinicalRecord
        fields = (
            "id",
            "patient",
            "recorded_by",
            "title",
            "clinical_notes",
            "recorded_at",
            "created_at",
        )

        read_only_fields = (
            "id",
            "recorded_by",
            "created_at",
        )


class PatientAssignmentSerializer(serializers.ModelSerializer):
    clinician_name = serializers.SerializerMethodField()

    class Meta:
        model = PatientAssignment
        fields = (
            "id",
            "patient",
            "clinician",
            "clinician_name",
            "is_primary",
            "assigned_by",
            "assigned_at",
        )

        read_only_fields = (
            "id",
            "clinician_name",
            "assigned_by",
            "assigned_at",
        )

    def get_clinician_name(self, obj):
        return f"{obj.clinician.first_name} " f"{obj.clinician.last_name}"

    def validate_clinician(self, value):
        if value.role != UserRole.CLINICIAN:
            raise serializers.ValidationError(
                "Only clinician accounts can be assigned to patients."
            )

        return value
