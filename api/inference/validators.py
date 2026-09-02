from rest_framework import serializers


def validate_inference_input(
    input_type,
    input_data,
):
    if not isinstance(input_data, dict):
        raise serializers.ValidationError(
            {"input_data": ("Inference input must be an object.")}
        )

    if input_type == "TABULAR":
        return validate_tabular_input(input_data)

    if input_type == "IMAGE":
        return validate_image_input(input_data)

    if input_type == "MULTIMODAL":
        return validate_multimodal_input(input_data)

    raise serializers.ValidationError(
        {"input_type": ("Unsupported inference input type.")}
    )


def validate_tabular_input(data):
    if not data:
        raise serializers.ValidationError(
            {"input_data": ("Tabular input cannot be empty.")}
        )

    return data


def validate_image_input(data):
    if "file_ids" not in data:
        raise serializers.ValidationError(
            {"input_data": ("Image inference requires file_ids.")}
        )

    return data


def validate_multimodal_input(data):
    if "clinical_data" not in data and "file_ids" not in data:
        raise serializers.ValidationError(
            {"input_data": ("Multimodal inference requires " "clinical data or files.")}
        )

    return data
