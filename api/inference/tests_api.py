from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import APIClient

from accounts.models import User, UserRole
from patients.models import Patient, PatientAssignment

from .models import Inference, InferenceImage, InferenceStatus, InferenceType
from .serializers import CreateSymptomsInferenceSerializer


class CreateSymptomsInferenceSerializerTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(
            email="owner@example.com",
            password="Password123!",
            first_name="Owner",
            last_name="Clinician",
            role=UserRole.CLINICIAN,
        )
        self.other = User.objects.create_user(
            email="other@example.com",
            password="Password123!",
            first_name="Other",
            last_name="Clinician",
            role=UserRole.CLINICIAN,
        )
        self.admin = User.objects.create_user(
            email="admin@example.com",
            password="Password123!",
            first_name="Admin",
            last_name="User",
            role=UserRole.ADMIN,
        )
        self.patient = Patient.objects.create(
            first_name="Jane",
            last_name="Doe",
            date_of_birth="1990-01-15",
            sex=Patient.Sex.FEMALE,
            email="jane@example.com",
            created_by=self.owner,
        )

    def test_serializer_accepts_flat_symptoms_payload(self):
        serializer = CreateSymptomsInferenceSerializer(
            data={
                "patient_id": self.patient.patient_id,
                "clinical_notes": "Needs review",
                "symptoms": {"age": 42, "glucose": 128, "bmi": 27.3},
            },
            context={"request": type("Request", (), {"user": self.owner})()},
        )

        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(
            serializer.validated_data["symptoms"],
            {"age": 42, "glucose": 128, "bmi": 27.3},
        )

    def test_serializer_rejects_missing_payload_without_crashing_when_request_context_is_absent(self):
        serializer = CreateSymptomsInferenceSerializer(
            data={"patient_id": self.patient.patient_id},
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("symptoms", serializer.errors)

    def test_serializer_rejects_access_for_unassigned_clinician(self):
        serializer = CreateSymptomsInferenceSerializer(
            data={
                "patient_id": self.patient.patient_id,
                "symptoms": {"age": 42},
            },
            context={"request": type("Request", (), {"user": self.other})()},
        )

        self.assertFalse(serializer.is_valid())
        self.assertIn("patient_id", serializer.errors)

    def test_serializer_accepts_legacy_models_payload_for_backwards_compatibility(self):
        serializer = CreateSymptomsInferenceSerializer(
            data={
                "patient_id": self.patient.patient_id,
                "models": {"heart_disease": {"age": 58, "cp": 1}},
            },
            context={"request": type("Request", (), {"user": self.owner})()},
        )

        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(
            serializer.validated_data["symptoms"],
            {"heart_disease": {"age": 58, "cp": 1}},
        )


class InferenceCreateViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.owner = User.objects.create_user(
            email="owner2@example.com",
            password="Password123!",
            first_name="Owner",
            last_name="Clinician",
            role=UserRole.CLINICIAN,
        )
        self.patient = Patient.objects.create(
            first_name="Alice",
            last_name="Brown",
            date_of_birth="1987-03-10",
            sex=Patient.Sex.FEMALE,
            email="alice@example.com",
            created_by=self.owner,
        )
        PatientAssignment.objects.create(
            patient=self.patient,
            clinician=self.owner,
            assigned_by=self.owner,
            is_primary=True,
        )

    def test_symptoms_create_endpoint_creates_inference_with_normalized_payload(self):
        self.client.force_authenticate(user=self.owner)

        response = self.client.post(
            "/api/v1/inference/symptoms/",
            {
                "patient_id": self.patient.patient_id,
                "clinical_notes": "Patient reports fatigue.",
                "symptoms": {"age": 46, "glucose": 140, "bmi": 30},
            },
            format="json",
        )

        self.assertEqual(response.status_code, 202)
        inference = Inference.objects.get(pk=response.data["id"])
        self.assertEqual(inference.inference_type, InferenceType.SYMPTOMS)
        self.assertEqual(
            inference.request_payload,
            {
                "symptoms": {"age": 46, "glucose": 140, "bmi": 30},
                "clinical_notes": "Patient reports fatigue.",
            },
        )
        self.assertEqual(inference.status, InferenceStatus.PENDING)

    def test_image_create_endpoint_creates_inference_and_attaches_file(self):
        self.client.force_authenticate(user=self.owner)

        image = SimpleUploadedFile(
            "scan.png",
            b"fake-image-bytes",
            content_type="image/png",
        )

        response = self.client.post(
            "/api/v1/inference/image/",
            {
                "patient_id": self.patient.patient_id,
                "image_type": "xray",
                "clinical_notes": "Chest X-ray review",
                "symptoms": '{"age": 51}',
                "file": image,
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, 202)
        inference = Inference.objects.get(pk=response.data["id"])
        self.assertEqual(inference.inference_type, InferenceType.IMAGE)
        self.assertEqual(inference.request_payload["symptoms"], {"age": 51})
        self.assertTrue(InferenceImage.objects.filter(inference=inference).exists())
