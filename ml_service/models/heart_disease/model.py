"""
models/heart_disease/model.py
Heart Disease classifier — XGBoost + Random Forest soft-voting ensemble.

Uses SoftVotingEnsemble (models/shared/ensemble.py) instead of sklearn's
VotingClassifier to avoid XGBoost 2.x sklearn compatibility issues.
"""
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from models.shared.ensemble import SoftVotingEnsemble

from core.logger import logger
from core.config import settings


CLASS_NAMES  = ["NO HEART DISEASE", "HEART DISEASE"]
NUM_CLASSES  = len(CLASS_NAMES)
IDX_TO_CLASS = {0: "NO HEART DISEASE", 1: "HEART DISEASE"}
CLASS_TO_IDX = {"NO HEART DISEASE": 0, "HEART DISEASE": 1}

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def build_model() -> Pipeline:
    xgb = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", device="cpu",
        random_state=42, n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=250, max_depth=8, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    ensemble = SoftVotingEnsemble(
        estimators=[("xgb", xgb), ("rf", rf)],
        weights=[2, 1],
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   ensemble),
    ])
    logger.info("Built Heart Disease Pipeline: Imputer → Scaler → SoftVotingEnsemble(XGB+RF)")
    return pipeline


def save_model(pipeline: Pipeline, path: Path | None = None):
    path = path or (settings.model_registry_dir / "heart_disease" / "best_model.joblib")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Heart Disease model saved → {path}")


def load_trained_model(path: Path | None = None) -> Pipeline | None:
    path = path or (settings.model_registry_dir / "heart_disease" / "best_model.joblib")
    if Path(path).exists():
        pipeline = joblib.load(path)
        logger.info(f"Loaded heart disease model from {path}")
        return pipeline
    logger.warning(
        f"No trained heart disease model found at {path}. "
        "Run training first: python -m training.heart_disease.train"
    )
    return None
