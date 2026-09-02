"""
models/shared/ensemble.py
Custom soft-voting ensemble compatible with XGBoost 2.x and sklearn scorers.

WHY THIS EXISTS:
  XGBoost 2.x no longer sets the 'estimator_type' tag correctly, causing
  sklearn's VotingClassifier to reject it: "XGBClassifier should be a classifier."

  Additionally, sklearn's built-in 'roc_auc' scorer for cross_validate calls
  predict_proba and expects a 1D array for binary classification (positive-class
  probabilities only). This ensemble's predict_proba returns (N, 2) for
  compatibility with inference, and a separate predict_proba_binary() method
  returns the 1D positive-class column for use with the custom roc_auc scorer.

USED BY:
  All 5 tabular disease models: Diabetes, Heart Disease, Anaemia,
  Hypertension, Hepatitis B.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted


class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Soft-voting ensemble: fits each estimator independently and
    returns the weighted average of their predict_proba() outputs.

    Drop-in replacement for VotingClassifier(voting='soft') that
    works correctly with XGBoost 2.x inside sklearn Pipelines.

    Args:
        estimators: list of (name, estimator) tuples
        weights:    list of floats, one per estimator (default: equal weight)
    """

    def __init__(self, estimators: list, weights: list = None):
        self.estimators = estimators
        self.weights    = weights

    def fit(self, X, y):
        self.classes_    = np.unique(y)
        self.n_classes_  = len(self.classes_)
        self.estimators_ = []

        w = self.weights if self.weights else [1.0] * len(self.estimators)
        self.weights_ = np.array(w, dtype=float) / np.sum(w)

        for _name, est in self.estimators:
            fitted = clone(est).fit(X, y)
            self.estimators_.append(fitted)

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Returns (N, n_classes) probability array — used for inference."""
        check_is_fitted(self, "estimators_")
        avg = np.zeros((X.shape[0], self.n_classes_))
        for fitted_est, w in zip(self.estimators_, self.weights_):
            avg += w * fitted_est.predict_proba(X)
        return avg

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_params(self, deep: bool = True) -> dict:
        params = {"estimators": self.estimators, "weights": self.weights}
        if deep:
            for name, est in self.estimators:
                for k, v in est.get_params(deep=True).items():
                    params[f"{name}__{k}"] = v
        return params
