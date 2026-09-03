from django.urls import path

from .views import (
    PatientListCreateView,
    PatientDetailView,
    ClinicalRecordListCreateView,
    PatientAssignmentCreateView,
    PatientAssignmentListView,
)

urlpatterns = [
    path(
        "",
        PatientListCreateView.as_view(),
        name="patient-list-create",
    ),
    path(
        "<str:patient_id>/",
        PatientDetailView.as_view(),
        name="patient-detail",
    ),
    path(
        "<int:patient_id>/records/",
        ClinicalRecordListCreateView.as_view(),
        name="patient-records",
    ),
    path(
        "<int:patient_id>/assignments/",
        PatientAssignmentListView.as_view(),
        name="patient-assignments",
    ),
    path(
        "assignments/",
        PatientAssignmentCreateView.as_view(),
        name="patient-assignment-create",
    ),
]
