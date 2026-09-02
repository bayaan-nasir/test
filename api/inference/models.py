from django.conf import settings
from django.db import models


class InferenceType(models.TextChoices):
    SYMPTOMS = "SYMPTOMS", "Symptoms"
    IMAGE = "IMAGE", "Image"


class InferenceStatus(models.TextChoices):
    PENDING = "PENDING", "Pending"
    PROCESSING = "PROCESSING", "Processing"
    COMPLETED = "COMPLETED", "Completed"
    FAILED = "FAILED", "Failed"


class Inference(models.Model):
    patient = models.ForeignKey(
        "patients.Patient",
        on_delete=models.PROTECT,
        related_name="inferences",
        null=True,
        blank=True,
    )

    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="requested_inferences",
    )

    inference_type = models.CharField(
        max_length=20,
        choices=InferenceType.choices,
    )

    status = models.CharField(
        max_length=20,
        choices=InferenceStatus.choices,
        default=InferenceStatus.PENDING,
    )

    request_payload = models.JSONField(
        default=dict,
    )

    response_payload = models.JSONField(
        default=dict,
    )

    fastapi_request_id = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        db_index=True,
    )

    overall_triage = models.CharField(
        max_length=20,
        blank=True,
    )

    predicted_class = models.CharField(
        max_length=255,
        blank=True,
    )

    confidence = models.FloatField(
        null=True,
        blank=True,
    )

    clinical_summary = models.TextField(
        blank=True,
    )

    error_message = models.TextField(
        blank=True,
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    started_at = models.DateTimeField(
        null=True,
        blank=True,
    )

    completed_at = models.DateTimeField(
        null=True,
        blank=True,
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["requested_by", "-created_at"]),
            models.Index(fields=["patient", "-created_at"]),
            models.Index(fields=["status", "-created_at"]),
            models.Index(fields=["inference_type", "-created_at"]),
        ]

    def __str__(self):
        return f"Inference #{self.pk}"


class InferenceImage(models.Model):
    inference = models.ForeignKey(
        Inference,
        on_delete=models.CASCADE,
        related_name="images",
    )

    file = models.FileField(
        upload_to="inferences/%Y/%m/%d/",
    )

    image_type = models.CharField(
        max_length=50,
    )

    clinical_notes = models.TextField(
        blank=True,
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    def __str__(self):
        return f"{self.image_type} - {self.file.name}"
