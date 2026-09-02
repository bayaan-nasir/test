"""
core/config.py
Loads all environment variables into a typed Settings object.
Import `settings` anywhere in the project — never read os.environ directly.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # Service 
    ml_service_host: str = Field("0.0.0.0", env="ML_SERVICE_HOST")
    ml_service_port: int = Field(8001, env="ML_SERVICE_PORT")
    debug: bool = Field(True, env="DEBUG")

    # Model paths 
    model_registry_dir: Path = Field("./model_registry", env="MODEL_REGISTRY_DIR")
    pneumonia_model_path: Path = Field(
        "./model_registry/pneumonia/best_model.pth",
        env="PNEUMONIA_MODEL_PATH"
    )

    # Inference
    confidence_threshold_high: float = Field(0.85, env="CONFIDENCE_THRESHOLD_HIGH")
    confidence_threshold_medium: float = Field(0.60, env="CONFIDENCE_THRESHOLD_MEDIUM")
    image_size: int = Field(224, env="IMAGE_SIZE")
    device: str = Field("cpu", env="DEVICE")

    #Media 
    gradcam_output_dir: Path = Field("./media/gradcam", env="GRADCAM_OUTPUT_DIR")
    upload_dir: Path = Field("./media/uploads", env="UPLOAD_DIR")

    #  MLflow 
    mlflow_tracking_uri: str = Field("./mlruns", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field("ghana_health_ai", env="MLFLOW_EXPERIMENT_NAME")

    # CORS 
    django_backend_url: str = Field("http://localhost:8000", env="DJANGO_BACKEND_URL")
    allowed_origins: str = Field(
        "http://localhost:3000,http://localhost:8000",
        env="ALLOWED_ORIGINS"
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    # Gemini LLM 
    gemini_api_key: str = Field("", env="GEMINI_API_KEY")
    gemini_model: str   = Field("gemini-3.6-flash", env="GEMINI_MODEL")

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

# Ensure media directories exist at startup
settings.gradcam_output_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.model_registry_dir.mkdir(parents=True, exist_ok=True)
