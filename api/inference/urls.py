from django.urls import path

from .views import (
    ImageInferenceCreateView,
    InferenceDetailView,
    InferenceListView,
    SymptomsInferenceCreateView,
)

urlpatterns = [
    path(
        "",
        InferenceListView.as_view(),
        name="inference-list",
    ),
    path(
        "symptoms/",
        SymptomsInferenceCreateView.as_view(),
        name="inference-symptoms-create",
    ),
    path(
        "image/",
        ImageInferenceCreateView.as_view(),
        name="inference-image-create",
    ),
    path(
        "<int:pk>/",
        InferenceDetailView.as_view(),
        name="inference-detail",
    ),
]
