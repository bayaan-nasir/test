"""
explainability/shap_explainer.py
SHAP-based explainability for tabular disease models (Diabetes, Cardiovascular, etc.)

For image models we use Grad-CAM. For tabular models we use SHAP TreeExplainer.

SHAP (SHapley Additive exPlanations) tells us:
  - Which features pushed the prediction toward DIABETIC vs NON-DIABETIC
  - The magnitude of each feature's contribution
  - The direction (positive = toward positive class, negative = away)

Output:
  - A list of top features with their SHAP values and actual input values
  - This is what Django receives as the `explainability.top_features` field
  - The frontend renders this as a horizontal bar chart
"""
import numpy as np
import shap
from sklearn.pipeline import Pipeline
from core.logger import logger


def compute_shap_for_prediction(
    pipeline: Pipeline,
    X_input: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
) -> list[dict]:
    """
    Compute SHAP values for a single prediction and return the top_n
    most influential features.

    Args:
        pipeline:      trained sklearn/imblearn Pipeline
        X_input:       (1, n_features) numpy array — the patient's input
        feature_names: list of feature name strings in column order
        top_n:         number of top features to return (default 5)

    Returns:
        list of dicts, each with keys:
          feature, value, shap, direction, rank
    """
    try:
        # Transform through imputer + scaler only — not SMOTE (inference only)
        X_transformed = pipeline.named_steps["imputer"].transform(X_input)
        X_transformed = pipeline.named_steps["scaler"].transform(X_transformed)

        # The "model" step might be an ensemble OR a direct tree classifier
        classifier = pipeline.named_steps["model"]
        
        # Dynamically extract the base tree model for SHAP
        if hasattr(classifier, "estimators_"):
            # Fitted sklearn ensemble
            tree_model = classifier.estimators_[0]
        elif hasattr(classifier, "estimators") and isinstance(classifier.estimators, list):
            # Custom ensemble (like SoftVotingEnsemble)
            if isinstance(classifier.estimators[0], tuple):
                tree_model = classifier.estimators[0][1]
            else:
                tree_model = classifier.estimators[0]
        else:
            # It's already a single tree model (XGBClassifier, RandomForest, etc.)
            tree_model = classifier

        explainer   = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_transformed)

        # shap_values shape: (1, n_features) for binary XGBoost
        # For multi-output models (like RF) it may be a list or a 3D array
        if isinstance(shap_values, list):
            sv = shap_values[1][0]   # SHAP for positive class, first sample
        elif len(np.array(shap_values).shape) == 3:
            sv = np.array(shap_values)[0, :, 1] # SHAP for positive class
        else:
            sv = shap_values[0]      # shape (n_features,)

        actual_values = X_input[0]   # original (pre-transform) values

        # Build feature list sorted by absolute SHAP magnitude
        features = []
        for i, (fname, sval, aval) in enumerate(
            zip(feature_names, sv, actual_values)
        ):
            features.append({
                "feature":   fname,
                "value":     float(round(aval, 3)) if not np.isnan(aval) else None,
                "shap":      float(round(sval, 4)),
                "direction": "positive" if sval > 0 else "negative",
            })

        # Sort by absolute SHAP value descending
        features.sort(key=lambda x: abs(x["shap"]), reverse=True)

        # Add rank
        for rank, feat in enumerate(features[:top_n], start=1):
            feat["rank"] = rank

        return features[:top_n]

    except Exception as e:
        logger.warning(f"SHAP computation failed during inference: {e}")
        # Return empty list — prediction still proceeds, just without SHAP
        return []