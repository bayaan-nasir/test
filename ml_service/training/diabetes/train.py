# Training pipeline for the Type 2 Diabetes classifier.
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score,
    accuracy_score, make_scorer
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from models.diabetes.model import (
    CLASS_NAMES, FEATURE_NAMES, save_model
)
from preprocessing.tabular_transforms import load_diabetes_dataframe
from core.config import settings
from core.logger import logger


DATA_PATH = Path("data/raw/diabetes/diabetes.csv")

HPARAMS = {
    "xgb_n_estimators":  300,
    "xgb_max_depth":     5,
    "xgb_learning_rate": 0.05,
    "xgb_subsample":     0.8,
    "xgb_colsample":     0.8,
    "n_folds":           5,
    "smote_k_neighbors": 5,
    "random_state":      42,
}


def roc_auc_proba_scorer(estimator, X, y):
    """Custom AUC scorer compatible with sklearn 1.9 — slices proba column 1 explicitly."""
    y_proba = estimator.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_proba)


def build_pipeline(params: dict) -> ImbPipeline:
    """
    Build the full imbalanced-learn Pipeline:
      Imputer → Scaler → SMOTE → XGBClassifier

    ImbPipeline applies SMOTE only during fit — not predict.
    This is critical: oversampling should never affect the test set.
    """
    xgb = XGBClassifier(
        n_estimators=params["xgb_n_estimators"],
        max_depth=params["xgb_max_depth"],
        learning_rate=params["xgb_learning_rate"],
        subsample=params["xgb_subsample"],
        colsample_bytree=params["xgb_colsample"],
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=params["random_state"],
        n_jobs=-1,
    )

    pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("smote",   SMOTE(
            k_neighbors=params["smote_k_neighbors"],
            random_state=params["random_state"]
        )),
        ("model",   xgb),
    ])

    return pipeline


def tune_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 30) -> dict:
    """
    Optuna hyperparameter search for XGBoost parameters.
    Optimises AUC-ROC via 3-fold cross-validation.
    """
    logger.info(f"Running Optuna hyperparameter search ({n_trials} trials)...")

    def objective(trial):
        params = {
            **HPARAMS,
            "xgb_n_estimators":  trial.suggest_int("xgb_n_estimators", 100, 500),
            "xgb_max_depth":     trial.suggest_int("xgb_max_depth", 3, 8),
            "xgb_learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True),
            "xgb_subsample":     trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "xgb_colsample":     trial.suggest_float("xgb_colsample", 0.6, 1.0),
        }
        pipeline = build_pipeline(params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_validate(
            pipeline, X, y, cv=cv,
            scoring=roc_auc_proba_scorer, n_jobs=-1
        )
        return scores["test_score"].mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {**HPARAMS, **study.best_params}
    logger.info(f"Best AUC-ROC from tuning: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return best


def evaluate_pipeline(pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute all evaluation metrics on the test set."""
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
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
    return metrics


def compute_shap_values(pipeline, X_test: np.ndarray) -> np.ndarray:
    """
    Compute SHAP values for the XGBoost model.
    Used to validate that the model focuses on clinically meaningful features.
    """
    try:
        xgb_model = pipeline.named_steps["model"]
        X_transformed = pipeline.named_steps["scaler"].transform(
            pipeline.named_steps["imputer"].transform(X_test)
        )
        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_transformed)
        logger.info("SHAP values computed successfully")
        return shap_values
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def train():
    if not DATA_PATH.exists():
        logger.error(
            f"Dataset not found at {DATA_PATH}. "
            "Run: python scripts\\download_diabetes_dataset.py"
        )
        return

    df = load_diabetes_dataframe(str(DATA_PATH))
    X  = df[FEATURE_NAMES].values.astype(np.float32)
    y  = df["target"].values.astype(int)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=HPARAMS["random_state"]
    )

    logger.info(
        f"Dataset: {len(X)} total | "
        f"{len(X_train)} train | {len(X_test)} test | "
        f"class ratio: {y.mean():.2f}"
    )

    best_params = tune_hyperparameters(X_train, y_train, n_trials=30)

    logger.info(f"Running {HPARAMS['n_folds']}-fold cross-validation...")
    cv_pipeline = build_pipeline(best_params)
    cv = StratifiedKFold(
        n_splits=HPARAMS["n_folds"], shuffle=True,
        random_state=HPARAMS["random_state"]
    )
    scoring = {
        "roc_auc":  roc_auc_proba_scorer,
        "f1":       make_scorer(f1_score),
        "accuracy": make_scorer(accuracy_score),
    }

    cv_results = cross_validate(
        cv_pipeline, X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    logger.info(
        f"CV AUC-ROC:  {cv_results['test_roc_auc'].mean():.4f} "
        f"± {cv_results['test_roc_auc'].std():.4f}"
    )
    logger.info(
        f"CV F1:       {cv_results['test_f1'].mean():.4f} "
        f"± {cv_results['test_f1'].std():.4f}"
    )
    logger.info(
        f"CV Accuracy: {cv_results['test_accuracy'].mean():.4f} "
        f"± {cv_results['test_accuracy'].std():.4f}"
    )

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
        feature_importance = dict(zip(FEATURE_NAMES, mean_shap.tolist()))
        logger.info(f"SHAP feature importance: {feature_importance}")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="diabetes_xgb"):
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("classes", CLASS_NAMES)
        mlflow.log_param("features", FEATURE_NAMES)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test",  len(X_test))

        mlflow.log_metrics({
            "cv_auc_mean":    cv_results["test_roc_auc"].mean(),
            "cv_auc_std":     cv_results["test_roc_auc"].std(),
            "cv_f1_mean":     cv_results["test_f1"].mean(),
            "test_auc":       test_metrics["auc_roc"],
            "test_f1":        test_metrics["f1"],
            "test_accuracy":  test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
        })

        mlflow.sklearn.log_model(final_pipeline, "diabetes_model")

    save_model(final_pipeline)
    logger.info("Diabetes training complete.")


if __name__ == "__main__":
    train()