from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from django.db import transaction

from rest_framework import serializers

from .models import (
    ClinicianProfile,
    ProfessionalRole,
    User,
    UserRole,
)


class UserSerializer(serializers.ModelSerializer):
    professional_role = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = (
            "id",
            "email",
            "first_name",
            "last_name",
            "phone_number",
            "role",
            "professional_role",
        )
        read_only_fields = fields

    def get_professional_role(self, obj):
        try:
            return obj.clinician_profile.professional_role
        except ClinicianProfile.DoesNotExist:
            return None


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True,
        validators=[validate_password],
    )

    confirm_password = serializers.CharField(
        write_only=True,
    )

    professional_role = serializers.ChoiceField(
        choices=ProfessionalRole.choices,
        write_only=True,
        required=True,
    )

    class Meta:
        model = User
        fields = (
            "email",
            "first_name",
            "last_name",
            "phone_number",
            "professional_role",
            "password",
            "confirm_password",
        )
        extra_kwargs = {
            "phone_number": {
                "required": True,
            },
        }

    def validate_email(self, value):
        if User.objects.filter(
            email__iexact=value
        ).exists():
            raise serializers.ValidationError(
                "An account with this email already exists."
            )

        return value.lower()

    def validate(self, attrs):
        if (
            attrs["password"]
            != attrs["confirm_password"]
        ):
            raise serializers.ValidationError(
                {
                    "confirm_password": (
                        "Passwords do not match."
                    )
                }
            )

        if not attrs.get("professional_role"):
            raise serializers.ValidationError(
                {
                    "professional_role": (
                        "Professional role is required."
                    )
                }
            )

        return attrs

    @transaction.atomic
    def create(self, validated_data):
        validated_data.pop(
            "confirm_password"
        )

        professional_role = (
            validated_data.pop(
                "professional_role"
            )
        )

        user = User.objects.create_user(
            role=UserRole.CLINICIAN,
            **validated_data,
        )

        ClinicianProfile.objects.create(
            user=user,
            professional_role=professional_role,
        )

        return user


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(
        write_only=True,
    )

    def validate(self, attrs):
        user = authenticate(
            request=self.context.get("request"),
            email=attrs["email"],
            password=attrs["password"],
        )

        if user is None:
            raise serializers.ValidationError(
                "Invalid email or password."
            )

        if not user.is_active:
            raise serializers.ValidationError(
                "This account is inactive."
            )

        attrs["user"] = user

        return attrs