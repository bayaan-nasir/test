"""
models/hypertension/model.py
Hypertension classifier — XGBoost + Random Forest soft-voting ensemble.

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


CLASS_NAMES  = ["NO HYPERTENSION", "HYPERTENSION"]
NUM_CLASSES  = len(CLASS_NAMES)
IDX_TO_CLASS = {0: "NO HYPERTENSION", 1: "HYPERTENSION"}
CLASS_TO_IDX = {"NO HYPERTENSION": 0, "HYPERTENSION": 1}

FEATURE_NAMES = [
    'Age', 'Sex', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
    'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'PhysHlth', 
    'DiffWalk', 'Stroke', 'Diabetes'
]


def build_model() -> Pipeline:
    xgb = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.06,
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
    logger.info("Built Hypertension Pipeline: Imputer → Scaler → SoftVotingEnsemble(XGB+RF)")
    return pipeline


def save_model(pipeline: Pipeline, path: Path | None = None):
    path = path or (settings.model_registry_dir / "hypertension" / "best_model.joblib")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Hypertension model saved → {path}")


def load_trained_model(path: Path | None = None) -> Pipeline | None:
    path = path or (settings.model_registry_dir / "hypertension" / "best_model.joblib")
    if Path(path).exists():
        pipeline = joblib.load(path)
        logger.info(f"Loaded hypertension model from {path}")
        return pipeline
    logger.warning(
        f"No trained hypertension model found at {path}. "
        "Run training first: python -m training.hypertension.train"
    )
    return None
