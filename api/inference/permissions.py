from rest_framework.permissions import BasePermission

from accounts.models import UserRole


class CanAccessInference(BasePermission):
    message = "You do not have access to this inference."

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

    def has_object_permission(self, request, view, obj):
        if request.user.role == UserRole.ADMIN:
            return True
        if obj.patient is None:
            return obj.requested_by_id == request.user.id
        return obj.patient.assignments.filter(clinician=request.user).exists()
