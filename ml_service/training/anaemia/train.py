"""
training/anaemia/train.py
Training pipeline for the Anaemia classifier.

Run from the project root (Windows CMD):
  python -m ml_service.training.anaemia.train
"""
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from models.shared.ensemble import SoftVotingEnsemble
from models.shared.scorers import roc_auc_scorer, SCORING
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from models.anaemia.model import CLASS_NAMES, FEATURE_NAMES, save_model
from preprocessing.tabular_transforms import load_anaemia_dataframe
from core.config import settings
from core.logger import logger


DATA_PATH = Path("data/raw/anaemia/anaemia.csv")

HPARAMS = {
    "xgb_n_estimators":  150,
    "xgb_max_depth":     3,
    "xgb_learning_rate": 0.08,
    "xgb_subsample":     0.8,
    "xgb_colsample":     0.8,
    "rf_n_estimators":   150,
    "rf_max_depth":      6,
    "n_folds":           5,
    "random_state":      42,
}


def build_pipeline(params: dict) -> Pipeline:
    xgb = XGBClassifier(
        n_estimators=params["xgb_n_estimators"],
        max_depth=params["xgb_max_depth"],
        learning_rate=params["xgb_learning_rate"],
        subsample=params["xgb_subsample"],
        colsample_bytree=params["xgb_colsample"],
        eval_metric="logloss",
        device="cpu",
        random_state=params["random_state"],
        n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=params["rf_n_estimators"],
        max_depth=params["rf_max_depth"],
        min_samples_split=4, min_samples_leaf=2,
        class_weight="balanced",
        random_state=params["random_state"], n_jobs=-1,
    )
    ensemble = SoftVotingEnsemble(estimators=[("xgb", xgb), ("rf", rf)], weights=[2, 1])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   ensemble),
    ])


def tune_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 20) -> dict:
    logger.info(f"Running Optuna hyperparameter search ({n_trials} trials)...")

    def objective(trial):
        params = {
            **HPARAMS,
            "xgb_n_estimators":  trial.suggest_int("xgb_n_estimators", 50, 250),
            "xgb_max_depth":     trial.suggest_int("xgb_max_depth", 2, 5),
            "xgb_learning_rate": trial.suggest_float("xgb_learning_rate", 0.02, 0.2, log=True),
            "xgb_subsample":     trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "xgb_colsample":     trial.suggest_float("xgb_colsample", 0.6, 1.0),
            "rf_n_estimators":   trial.suggest_int("rf_n_estimators", 50, 250),
            "rf_max_depth":      trial.suggest_int("rf_max_depth", 3, 10),
        }
        pipeline = build_pipeline(params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=roc_auc_scorer, n_jobs=1)
        return scores["test_score"].mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = {**HPARAMS, **study.best_params}
    logger.info(f"Best AUC-ROC from tuning: {study.best_value:.4f}")
    return best


def evaluate_pipeline(pipeline, X_test, y_test) -> dict:
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "auc_roc":   roc_auc_score(y_test, y_proba),
        "f1":        f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
        ),
    }


def compute_shap_values(pipeline, X_test):
    try:
        xgb_model = pipeline.named_steps["model"].estimators_[0]
        X_t = pipeline.named_steps["scaler"].transform(
            pipeline.named_steps["imputer"].transform(X_test)
        )
        return shap.TreeExplainer(xgb_model).shap_values(X_t)
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def train():
    if not DATA_PATH.exists():
        logger.error(f"Dataset not found at {DATA_PATH}. Run: python scripts\\download_anaemia_dataset.py")
        return

    df = load_anaemia_dataframe(str(DATA_PATH))
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["target"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=HPARAMS["random_state"]
    )
    logger.info(f"Dataset: {len(X)} total | {len(X_train)} train | {len(X_test)} test | class ratio: {y.mean():.2f}")

    best_params = tune_hyperparameters(X_train, y_train, n_trials=20)

    logger.info(f"Running {HPARAMS['n_folds']}-fold cross-validation...")
    cv_pipeline = build_pipeline(best_params)
    cv = StratifiedKFold(n_splits=HPARAMS["n_folds"], shuffle=True, random_state=HPARAMS["random_state"])
    cv_results = cross_validate(cv_pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
    logger.info(f"CV AUC-ROC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")

    logger.info("Training final model on full training set...")
    final_pipeline = build_pipeline(best_params)
    final_pipeline.fit(X_train, y_train)

    test_metrics = evaluate_pipeline(final_pipeline, X_test, y_test)
    logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f} | F1: {test_metrics['f1']:.4f} | Acc: {test_metrics['accuracy']:.4f}")
    logger.info(f"Confusion matrix:\n{test_metrics['confusion_matrix']}")

    shap_values = compute_shap_values(final_pipeline, X_test)
    if shap_values is not None:
        mean_shap = np.abs(shap_values).mean(axis=0)
        logger.info(f"SHAP feature importance: {dict(zip(FEATURE_NAMES, mean_shap.tolist()))}")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="anaemia_xgb_rf_ensemble"):
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "SoftVotingEnsemble(XGBoost + RandomForest)")
        mlflow.log_metrics({
            "cv_auc_mean": cv_results["test_roc_auc"].mean(),
            "test_auc": test_metrics["auc_roc"], "test_f1": test_metrics["f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"], "test_recall": test_metrics["recall"],
        })
        mlflow.sklearn.log_model(final_pipeline, "anaemia_model")

    save_model(final_pipeline)
    logger.info("Anaemia training complete.")


if __name__ == "__main__":
    train()
