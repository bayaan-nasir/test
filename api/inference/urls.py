from django.urls import path

from .views import (
    ImageInferenceCreateView,
    InferenceDetailView,
    InferenceListView,
    ModelRegistryView,
    ModelStatsView,
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
    path("models/", ModelRegistryView.as_view(), name="inference-models"),
    path("model-stats/", ModelStatsView.as_view(), name="inference-model-stats"),
]
