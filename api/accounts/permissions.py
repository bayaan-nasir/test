from rest_framework.permissions import BasePermission

from accounts.models import UserRole


class IsClinician(BasePermission):
    message = "Clinician access is required."

    def has_permission(self, request, view):
        return (
            request.user
            and request.user.is_authenticated
            and request.user.role == UserRole.CLINICIAN
        )


class IsAdmin(BasePermission):
    message = "Administrator access is required."

    def has_permission(self, request, view):
        return (
            request.user
            and request.user.is_authenticated
            and request.user.role == UserRole.ADMIN
        )


class IsClinicianOrAdmin(BasePermission):
    message = "Clinician or administrator access is required."

    def has_permission(self, request, view):
        return (
            request.user
            and request.user.is_authenticated
            and request.user.role
            in {
                UserRole.CLINICIAN,
                UserRole.ADMIN,
            }
        )


class IsPatient(BasePermission):
    message = "Patient access is required."

    def has_permission(self, request, view):
        return (
            request.user
            and request.user.is_authenticated
            and request.user.role == UserRole.PATIENT
        )
