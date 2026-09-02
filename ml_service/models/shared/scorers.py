"""
models/shared/scorers.py
Custom sklearn scorers for use with cross_validate() across all tabular models.

WHY THIS EXISTS:
  sklearn's built-in "roc_auc" scorer calls predict_proba(X) and expects a
  1D array of positive-class probabilities for binary classification.
  Our SoftVotingEnsemble correctly returns (N, 2) for inference compatibility,
  so the built-in scorer breaks with:
    "ValueError: y should be a 1d array, got an array of shape (N, 2)"

  The fix: use make_scorer() with a wrapper that slices [:, 1] from the
  2-column output before passing it to roc_auc_score.

USAGE in train.py:
  from models.shared.scorers import roc_auc_scorer, SCORING

  # In Optuna objective:
  scores = cross_validate(pipeline, X, y, cv=cv, scoring=roc_auc_scorer, n_jobs=1)

  # In final CV:
  cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
"""
import numpy as np
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, accuracy_score


def _roc_auc_binary(y_true, y_proba):
    """
    Wrapper that handles both 1D and 2D predict_proba output.
    For binary classification, extracts the positive-class (index 1) column.
    """
    if hasattr(y_proba, "ndim") and y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    return roc_auc_score(y_true, y_proba)


# Single scorer for Optuna objective (response_method handles the probability mapping)
roc_auc_scorer = make_scorer(
    _roc_auc_binary,
    response_method="predict_proba",
)

# Multi-metric dict for final cross-validation run
SCORING = {
    "roc_auc":  roc_auc_scorer,
    "f1":       make_scorer(f1_score),
    "accuracy": make_scorer(accuracy_score),
}
