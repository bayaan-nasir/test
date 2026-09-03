from django.db.models.deletion import ProtectedError
from django.db.models import Q
from rest_framework import generics, status
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from accounts.models import UserRole
from accounts.permissions import IsClinicianOrAdmin, IsAdmin

from .models import Patient, ClinicalRecord, PatientAssignment
from .serializers import (
    PatientAssignmentSerializer,
    PatientSerializer,
    ClinicalRecordSerializer,
)


class PatientListCreateView(generics.ListCreateAPIView):
    serializer_class = PatientSerializer
    permission_classes = [
        IsClinicianOrAdmin,
    ]

    def get_queryset(self):
        user = self.request.user

        if user.role == UserRole.ADMIN:
            queryset = Patient.objects.all()
        else:
            queryset = Patient.objects.filter(assignments__clinician=user).distinct()

        search = self.request.query_params.get("search", "").strip()

        if search:
            queryset = queryset.filter(
                Q(patient_id__icontains=search)
                | Q(first_name__icontains=search)
                | Q(last_name__icontains=search)
                | Q(phone_number__icontains=search)
                | Q(email__icontains=search)
            )

        return queryset

    def perform_create(self, serializer):
        patient = serializer.save(created_by=self.request.user)

        # A newly created patient must be assigned to its creating clinician,
        # otherwise get_queryset() (which filters on `assignments__clinician`)
        # will never return it back to them.
        if self.request.user.role == UserRole.CLINICIAN:
            PatientAssignment.objects.get_or_create(
                patient=patient,
                clinician=self.request.user,
                defaults={
                    "assigned_by": self.request.user,
                    "is_primary": True,
                },
            )


class PatientDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = PatientSerializer
    permission_classes = [
        IsClinicianOrAdmin,
    ]
    lookup_field = "patient_id"
    lookup_url_kwarg = "patient_id"

    def get_queryset(self):
        user = self.request.user

        if user.role == UserRole.ADMIN:
            return Patient.objects.all()

        return Patient.objects.filter(assignments__clinician=user).distinct()

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        try:
            instance.delete()
        except ProtectedError:
            return Response(
                {
                    "detail": (
                        "This patient has existing AI inference records and "
                        "cannot be deleted."
                    )
                },
                status=status.HTTP_409_CONFLICT,
            )
        return Response(status=status.HTTP_204_NO_CONTENT)


class ClinicalRecordListCreateView(generics.ListCreateAPIView):
    serializer_class = ClinicalRecordSerializer
    permission_classes = [
        IsClinicianOrAdmin,
    ]

    def get_patient(self):
        user = self.request.user
        patient_id = self.kwargs["patient_id"]

        queryset = Patient.objects.filter(pk=patient_id)

        if user.role != UserRole.ADMIN:
            queryset = queryset.filter(assignments__clinician=user)

        return get_object_or_404(queryset.distinct())

    def get_queryset(self):
        patient = self.get_patient()

        return ClinicalRecord.objects.filter(patient=patient)

    def perform_create(self, serializer):
        patient = self.get_patient()

        serializer.save(
            patient=patient,
            recorded_by=self.request.user,
        )


class PatientAssignmentCreateView(generics.CreateAPIView):
    serializer_class = PatientAssignmentSerializer
    permission_classes = [
        IsAdmin,
    ]

    def perform_create(self, serializer):
        serializer.save(assigned_by=self.request.user)


class PatientAssignmentListView(generics.ListAPIView):
    serializer_class = PatientAssignmentSerializer
    permission_classes = [
        IsClinicianOrAdmin,
    ]

    def get_queryset(self):
        patient_id = self.kwargs["patient_id"]

        return PatientAssignment.objects.filter(patient_id=patient_id).select_related(
            "clinician",
            "assigned_by",
        )
