"""Prediction and batch scoring pipeline using loaded model bundle."""

from pathlib import Path

import pandas as pd

from turbineguard.artifacts import LoadedBundle
from turbineguard.data import (
    RAW_COLUMNS,
    ValidationError,
    load_raw_cmapss_file,
    validate_finite_values,
    validate_trajectory_chronology,
)
from turbineguard.features import extract_engine_features_at_cycle
from turbineguard.policy import compute_inspection_priority, is_within_horizon, rank_worklist


def load_and_validate_input_file(file_path: Path, input_format: str = "cmapss") -> pd.DataFrame:
    """Load and validate batch input data in either CMAPSS or headered CSV format."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if input_format == "cmapss":
        df = load_raw_cmapss_file(file_path, partition="batch_input")
    elif input_format == "csv":
        df = pd.read_csv(file_path)
        missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValidationError(f"CSV input missing required columns: {missing_cols}")
        df["source_partition"] = "batch_input"
    else:
        raise ValueError(f"Unsupported input format '{input_format}'. Use 'cmapss' or 'csv'.")

    # Validate data integrity
    validate_finite_values(df, "batch_input")
    validate_trajectory_chronology(df, "batch_input")
    return df


def score_batch_history(
    bundle: LoadedBundle,
    input_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score latest cycle of each engine in input_df:
    - Extracts 20-cycle causal features at engine's max cycle.
    - Computes estimated RUL, within_horizon, anomaly_score, anomaly_flag, priority.
    - Returns (ranked_worklist_df, latest_features_df).
    """
    rul_features = bundle.feature_schema["rul_feature_names"]
    ano_features = bundle.feature_schema["anomaly_feature_names"]
    cutoff = bundle.policy["anomaly_cutoff"]
    horizon = bundle.policy["horizon_cycles"]

    worklist_rows = []
    feature_rows = []

    for unit_id, group in input_df.groupby("unit_id", sort=True):
        latest_c = int(group["cycle"].max())
        
        # Extract features ending at latest_c
        feats = extract_engine_features_at_cycle(
            group,
            target_cycle=latest_c,
            window_size=bundle.feature_schema["window_size"],
        )
        feats["unit_id"] = int(unit_id)
        feats["latest_cycle"] = latest_c
        feature_rows.append(feats)

        # Predict RUL
        df_feat = pd.DataFrame([feats])
        X_rul = df_feat[rul_features]
        pred_raw = float(bundle.rul_pipeline.predict(X_rul)[0])
        est_rul = max(0.0, pred_raw)

        # Score Anomaly
        X_ano = df_feat[ano_features].values
        X_ano_scaled = bundle.anomaly_pipeline.named_steps["scaler"].transform(X_ano)
        raw_score = float(bundle.anomaly_pipeline.named_steps["model"].score_samples(X_ano_scaled)[0])
        ano_score = -raw_score
        ano_flag = bool(ano_score > cutoff)

        in_horizon = bool(is_within_horizon(est_rul, horizon))
        priority = compute_inspection_priority(in_horizon, ano_flag)

        worklist_rows.append({
            "unit_id": int(unit_id),
            "latest_cycle": latest_c,
            "estimated_rul": est_rul,
            "within_horizon": in_horizon,
            "anomaly_score": ano_score,
            "anomaly_flag": ano_flag,
            "priority": priority,
            "bundle_version": bundle.version,
        })

    worklist_df = pd.DataFrame(worklist_rows)
    latest_features_df = pd.DataFrame(feature_rows)

    # Rank worklist deterministically
    ranked_worklist = rank_worklist(
        worklist_df,
        rul_col="estimated_rul",
        anomaly_col="anomaly_flag",
        unit_col="unit_id",
    )
    return ranked_worklist, latest_features_df
