# Type 2 Diabetes Classifier
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from core.logger import logger
from core.config import settings


CLASS_NAMES  = ["NON-DIABETIC", "DIABETIC"]
NUM_CLASSES  = len(CLASS_NAMES)
IDX_TO_CLASS = {0: "NON-DIABETIC", 1: "DIABETIC"}
CLASS_TO_IDX = {"NON-DIABETIC": 0, "DIABETIC": 1}

# Input features — must match column order in the dataset
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def build_model() -> Pipeline:
    """
    Build the full sklearn Pipeline:
      Imputer → Scaler → VotingClassifier(XGBoost + RandomForest)

    The Pipeline is a single serialisable object — saved/loaded as one .joblib file.
    """
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",   # handles the ~65/35 class imbalance in PIMA
        random_state=42,
        n_jobs=-1,
    )

    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf)],
        voting="soft",    # average probabilities
        weights=[2, 1],   # XGBoost slightly higher weight (stronger on tabular)
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # handles 0s as missing values
        ("scaler",  StandardScaler()),
        ("model",   ensemble),
    ])

    logger.info("Built Diabetes Pipeline: Imputer → Scaler → XGB+RF Ensemble")
    return pipeline


def save_model(pipeline: Pipeline, path: Path | None = None):
    """Save the trained pipeline to a .joblib file."""
    path = path or (settings.model_registry_dir / "diabetes" / "best_model.joblib")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Diabetes model saved → {path}")


def load_trained_model(path: Path | None = None) -> Pipeline | None:
    """Load a trained diabetes pipeline from disk."""
    path = path or (settings.model_registry_dir / "diabetes" / "best_model.joblib")

    if Path(path).exists():
        pipeline = joblib.load(path)
        logger.info(f"Loaded diabetes model from {path}")
        return pipeline
    else:
        logger.warning(
            f"No trained diabetes model found at {path}. "
            "Run training first: python -m training.diabetes.train"
        )
        return None
