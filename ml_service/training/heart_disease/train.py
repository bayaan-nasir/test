"""
training/heart_disease/train.py
Training pipeline for the Heart Disease classifier.

Run from the project root (Windows CMD):
  python -m ml_service.training.heart_disease.train
"""
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
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
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from models.heart_disease.model import CLASS_NAMES, FEATURE_NAMES, save_model
from preprocessing.tabular_transforms import load_heart_disease_dataframe
from core.config import settings
from core.logger import logger

DATA_PATH = Path("data/raw/heart_disease/heart.csv")

HPARAMS = {
    "xgb_n_estimators":  250,
    "xgb_max_depth":     4,
    "xgb_learning_rate": 0.05,
    "xgb_subsample":     0.8,
    "xgb_colsample":     0.8,
    "rf_n_estimators":   250,
    "rf_max_depth":      8,
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
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=params["random_state"],
        n_jobs=-1,
    )
    ensemble = SoftVotingEnsemble(
        estimators=[("xgb", xgb), ("rf", rf)],
        weights=[2, 1],
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   ensemble),
    ])

def tune_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 30) -> dict:
    logger.info(f"Running Optuna hyperparameter search ({n_trials} trials)...")

    def objective(trial):
        params = {
            **HPARAMS,
            "xgb_n_estimators":  trial.suggest_int("xgb_n_estimators", 100, 400),
            "xgb_max_depth":     trial.suggest_int("xgb_max_depth", 2, 6),
            "xgb_learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True),
            "xgb_subsample":     trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "xgb_colsample":     trial.suggest_float("xgb_colsample", 0.6, 1.0),
            "rf_n_estimators":   trial.suggest_int("rf_n_estimators", 100, 400),
            "rf_max_depth":      trial.suggest_int("rf_max_depth", 4, 12),
        }
        pipeline = build_pipeline(params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # ── EXPLICIT LOOP: Bypasses Scikit-Learn 2D array crashing bugs ──
        fold_aucs = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            preds_proba = pipeline.predict_proba(X_va)[:, 1]
            fold_aucs.append(roc_auc_score(y_va, preds_proba))
            
        return np.mean(fold_aucs)

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

def compute_shap_values(pipeline, X_test) -> np.ndarray:
    try:
        xgb_model = pipeline.named_steps["model"].estimators_[0]
        X_transformed = pipeline.named_steps["scaler"].transform(
            pipeline.named_steps["imputer"].transform(X_test)
        )
        explainer = shap.TreeExplainer(xgb_model)
        return explainer.shap_values(X_transformed)
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None

def train():
    if not DATA_PATH.exists():
        logger.error(
            f"Dataset not found at {DATA_PATH}. "
            "Run: python scripts\\download_heart_disease_dataset.py"
        )
        return

    df = load_heart_disease_dataframe(str(DATA_PATH))
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["target"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=HPARAMS["random_state"]
    )
    logger.info(
        f"Dataset: {len(X)} total | {len(X_train)} train | {len(X_test)} test | "
        f"class ratio: {y.mean():.2f}"
    )

    best_params = tune_hyperparameters(X_train, y_train, n_trials=30)

    logger.info(f"Running {HPARAMS['n_folds']}-fold cross-validation...")
    cv_pipeline = build_pipeline(best_params)
    cv = StratifiedKFold(
        n_splits=HPARAMS["n_folds"], shuffle=True, random_state=HPARAMS["random_state"]
    )
    
    # ── EXPLICIT LOOP FOR FINAL CV EVALUATION ──
    cv_auc, cv_f1 = [], []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]
        
        cv_pipeline.fit(X_tr, y_tr)
        preds_proba = cv_pipeline.predict_proba(X_va)[:, 1]
        preds_class = cv_pipeline.predict(X_va)
        
        cv_auc.append(roc_auc_score(y_va, preds_proba))
        cv_f1.append(f1_score(y_va, preds_class))

    cv_auc_mean, cv_auc_std = np.mean(cv_auc), np.std(cv_auc)
    cv_f1_mean, cv_f1_std = np.mean(cv_f1), np.std(cv_f1)

    logger.info(f"CV AUC-ROC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
    logger.info(f"CV F1:      {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

    logger.info("Training final model on full training set...")
    final_pipeline = build_pipeline(best_params)
    final_pipeline.fit(X_train, y_train)

    test_metrics = evaluate_pipeline(final_pipeline, X_test, y_test)
    logger.info(f"Test AUC-ROC  : {test_metrics['auc_roc']:.4f}")
    logger.info(f"Test F1 Score : {test_metrics['f1']:.4f}")
    logger.info(f"Test Accuracy : {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall   : {test_metrics['recall']:.4f}")
    logger.info(f"Confusion matrix:\n{test_metrics['confusion_matrix']}")

    shap_values = compute_shap_values(final_pipeline, X_test)
    if shap_values is not None:
        mean_shap = np.abs(shap_values).mean(axis=0)
        logger.info(
            f"SHAP feature importance: {dict(zip(FEATURE_NAMES, mean_shap.tolist()))}"
        )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="heart_disease_xgb_rf_ensemble"):
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "SoftVotingEnsemble(XGBoost + RandomForest)")
        mlflow.log_param("classes", CLASS_NAMES)
        mlflow.log_param("features", FEATURE_NAMES)

        mlflow.log_metrics({
            "cv_auc_mean":    cv_auc_mean,
            "cv_auc_std":     cv_auc_std,
            "cv_f1_mean":     cv_f1_mean,
            "test_auc":       test_metrics["auc_roc"],
            "test_f1":        test_metrics["f1"],
            "test_accuracy":  test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
        })
        mlflow.sklearn.log_model(final_pipeline, "heart_disease_model")

    save_model(final_pipeline)
    logger.info("Heart Disease training complete.")

if __name__ == "__main__":
    train()