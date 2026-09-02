from django.conf import settings
from django.db import models
import secrets


def generate_patient_id():
    return f"PT-{secrets.token_hex(4).upper()}"


class Patient(models.Model):
    class Sex(models.TextChoices):
        MALE = "MALE", "Male"
        FEMALE = "FEMALE", "Female"
        OTHER = "OTHER", "Other"

    patient_id = models.CharField(
        max_length=32,
        unique=True,
        db_index=True,
        default=generate_patient_id,
        editable=False,
    )

    first_name = models.CharField(
        max_length=150,
    )

    last_name = models.CharField(
        max_length=150,
    )

    date_of_birth = models.DateField()

    sex = models.CharField(
        max_length=10,
        choices=Sex.choices,
    )

    phone_number = models.CharField(
        max_length=30,
        blank=True,
    )

    email = models.EmailField(
        blank=True,
    )

    notes = models.TextField(
        blank=True,
    )

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="created_patients",
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    updated_at = models.DateTimeField(
        auto_now=True,
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.first_name} {self.last_name}"


class PatientAssignment(models.Model):
    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name="assignments",
    )

    clinician = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="patient_assignments",
    )

    assigned_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="patient_assignments_created",
    )

    is_primary = models.BooleanField(
        default=False,
    )

    assigned_at = models.DateTimeField(
        auto_now_add=True,
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["patient", "clinician"],
                name="unique_patient_clinician_assignment",
            ),
        ]
        ordering = ["-assigned_at"]

    def __str__(self):
        return f"{self.clinician} → {self.patient}"


class ClinicalRecord(models.Model):
    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name="clinical_records",
    )

    recorded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="clinical_records",
    )

    title = models.CharField(
        max_length=255,
    )

    clinical_notes = models.TextField()

    recorded_at = models.DateTimeField()

    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    class Meta:
        ordering = ["-recorded_at"]

    def __str__(self):
        return f"{self.patient} - {self.title}"
