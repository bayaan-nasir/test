"""
models/hepatitis_b/model.py
Hepatitis B classifier — XGBoost + Random Forest soft-voting ensemble.

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


CLASS_NAMES  = ["LOW RISK", "HIGH RISK"]
NUM_CLASSES  = len(CLASS_NAMES)
IDX_TO_CLASS = {0: "LOW RISK", 1: "HIGH RISK"}
CLASS_TO_IDX = {"LOW RISK": 0, "HIGH RISK": 1}

FEATURE_NAMES = [
    "age", "sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia", 
    "liver_big", "liver_firm", "spleen_palpable", "spiders", "ascites", 
    "varices", "bilirubin", "alk_phosphate", "sgot", "albumin", 
    "protime", "histology"
]


def build_model() -> Pipeline:
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", device="cpu",
        random_state=42, n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=7, min_samples_split=4,
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
    logger.info("Built Hepatitis B Pipeline: Imputer → Scaler → SoftVotingEnsemble(XGB+RF)")
    return pipeline


def save_model(pipeline, path: Path | None = None):
    path = path or (settings.model_registry_dir / "hepatitis_b" / "best_model.joblib")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Hepatitis B model saved → {path}")


def load_trained_model(path: Path | None = None):
    path = path or (settings.model_registry_dir / "hepatitis_b" / "best_model.joblib")
    if Path(path).exists():
        pipeline = joblib.load(path)
        logger.info(f"Loaded hepatitis B model from {path}")
        return pipeline
    logger.warning(
        f"No trained hepatitis B model found at {path}. "
        "Run training first: python -m training.hepatitis_b.train"
    )
    return None
