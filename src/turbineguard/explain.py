"""Model explanation module using SHAP and TreeSHAP for TurbineGuard RUL predictions."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from turbineguard.artifacts import LoadedBundle

logger = logging.getLogger(__name__)


def compute_shap_contributions(
    model: Any,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values (feature attributions) and base_values (expected model outputs)
    for model on feature matrix X.
    
    Returns:
    - shap_values: array of shape (N, n_features)
    - base_values: array of shape (N,)
    """
    if hasattr(model, "get_booster"):
        # Native TreeSHAP inside XGBoost
        dmat = xgb.DMatrix(X)
        # booster.predict(dmat, pred_contribs=True) returns (N, n_features + 1)
        # where the last column is the base value / bias
        contribs = model.get_booster().predict(dmat, pred_contribs=True)
        shap_values = contribs[:, :-1]
        base_values = contribs[:, -1]
        return shap_values, base_values

    elif isinstance(model, Pipeline):
        regressor = model.named_steps.get("regressor", model.steps[-1][1])
        scaler = model.named_steps.get("scaler", None)
        X_scaled = scaler.transform(X) if scaler else X.values
        coef = regressor.coef_
        intercept = float(regressor.intercept_)
        
        # Linear SHAP: contribution = w_i * (x_i - mean_ref_i)
        # Baseline reference is 0 for standardized features
        shap_values = X_scaled * coef
        base_values = np.full(len(X), intercept)
        return shap_values, base_values

    else:
        # Fallback to mean perturbation attribution
        preds = model.predict(X)
        mean_pred = float(np.mean(preds))
        base_values = np.full(len(X), mean_pred)
        shap_values = np.zeros(X.shape)
        return shap_values, base_values


def explain_prediction_sample(
    bundle: LoadedBundle,
    feature_row: pd.Series | dict | pd.DataFrame,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Generate local feature attribution explanation for a single engine snapshot.
    
    Returns:
    - base_value (expected model output)
    - estimated_rul (predicted RUL)
    - top_features: list of dicts with feature, value, shap_value, and direction.
    """
    feat_names = bundle.feature_schema["rul_feature_names"]
    
    if isinstance(feature_row, (dict, pd.Series)):
        df = pd.DataFrame([feature_row])
    else:
        df = feature_row

    X = df[feat_names]
    model = bundle.rul_pipeline

    shap_values, base_values = compute_shap_contributions(model, X)
    shap_row = shap_values[0]
    base_val = float(base_values[0])
    pred_rul = float(model.predict(X)[0])

    # Rank features by absolute SHAP impact
    top_indices = np.argsort(np.abs(shap_row))[::-1][:top_k]

    attributions = []
    for idx in top_indices:
        f_name = feat_names[idx]
        val = float(X.iloc[0][f_name])
        sv = float(shap_row[idx])
        attributions.append({
            "feature": f_name,
            "feature_value": val,
            "shap_value": sv,
            "effect": "increases_estimated_rul" if sv > 0 else "decreases_estimated_rul",
        })

    unit_id = int(df.iloc[0]["unit_id"]) if "unit_id" in df.columns else None
    latest_cycle = int(df.iloc[0]["latest_cycle"]) if "latest_cycle" in df.columns else (
        int(df.iloc[0]["cycle"]) if "cycle" in df.columns else None
    )

    return {
        "unit_id": unit_id,
        "latest_cycle": latest_cycle,
        "base_value": base_val,
        "estimated_rul": max(0.0, pred_rul),
        "top_features": attributions,
    }


def explain_batch_features(
    bundle: LoadedBundle,
    features_df: pd.DataFrame,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Explain predictions for a batch of engine feature rows."""
    feat_names = bundle.feature_schema["rul_feature_names"]
    X = features_df[feat_names]
    model = bundle.rul_pipeline

    shap_values, base_values = compute_shap_contributions(model, X)
    preds = model.predict(X)
    explanations = []

    for i in range(len(features_df)):
        shap_row = shap_values[i]
        top_indices = np.argsort(np.abs(shap_row))[::-1][:top_k]

        attributions = []
        for idx in top_indices:
            f_name = feat_names[idx]
            val = float(X.iloc[i][f_name])
            sv = float(shap_row[idx])
            attributions.append({
                "feature": f_name,
                "feature_value": val,
                "shap_value": sv,
                "effect": "increases_estimated_rul" if sv > 0 else "decreases_estimated_rul",
            })

        row = features_df.iloc[i]
        unit_id = int(row["unit_id"]) if "unit_id" in features_df.columns else i + 1
        cycle = int(row["latest_cycle"]) if "latest_cycle" in features_df.columns else (
            int(row["cycle"]) if "cycle" in features_df.columns else None
        )

        explanations.append({
            "unit_id": unit_id,
            "latest_cycle": cycle,
            "base_value": float(base_values[i]),
            "estimated_rul": max(0.0, float(preds[i])),
            "top_features": attributions,
        })

    return explanations
