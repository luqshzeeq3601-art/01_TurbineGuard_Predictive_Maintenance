"""Shared causal feature extraction for TurbineGuard."""

import numpy as np
import pandas as pd

from turbineguard.data import (
    SENSOR_COLUMNS,
    ValidationError,
    validate_trajectory_chronology,
)

# Constants for 20-cycle rolling OLS slope
WINDOW_SIZE = 20
X_REL = np.arange(WINDOW_SIZE, dtype=float)  # 0.0, 1.0, ..., 19.0
X_CENTERED = X_REL - X_REL.mean()  # -9.5, -8.5, ..., 9.5
X_SUM_SQ = float(np.sum(X_CENTERED**2))  # 665.0


def compute_rolling_ols_slope(y_window: np.ndarray) -> float:
    """
    Compute OLS linear slope of 1D array of length 20 against relative steps 0..19.
    slope = sum((x - x_mean) * y) / sum((x - x_mean)^2)
    """
    if len(y_window) != WINDOW_SIZE:
        raise ValueError(f"Expected window size {WINDOW_SIZE}, got {len(y_window)}")
    return float(np.dot(X_CENTERED, y_window) / X_SUM_SQ)


def extract_engine_features_at_cycle(
    engine_df: pd.DataFrame,
    target_cycle: int,
    sensor_cols: list[str] = SENSOR_COLUMNS,
    window_size: int = WINDOW_SIZE,
) -> dict[str, float]:
    """
    Extract causal features for a single engine ending at target_cycle.
    Requires at least window_size consecutive cycles ending at target_cycle.
    """
    # Filter rows up to target_cycle
    hist = engine_df[engine_df["cycle"] <= target_cycle].sort_values("cycle")
    if len(hist) < window_size:
        raise ValidationError(
            f"Insufficient history: required {window_size} cycles ending at cycle {target_cycle}, got {len(hist)}"
        )
    
    # Take the last window_size rows
    window = hist.iloc[-window_size:]
    
    # Check that cycles in the window are consecutive ending at target_cycle
    cycles = window["cycle"].values
    if cycles[-1] != target_cycle:
        raise ValidationError(f"Window latest cycle {cycles[-1]} does not match target cycle {target_cycle}")
    if not np.all(np.diff(cycles) == 1):
        raise ValidationError(f"Window cycles ending at {target_cycle} are not strictly consecutive.")

    features: dict[str, float] = {
        "cycle": float(target_cycle),
        "op_1": float(window["op_1"].iloc[-1]),
        "op_2": float(window["op_2"].iloc[-1]),
        "op_3": float(window["op_3"].iloc[-1]),
    }

    for sensor in sensor_cols:
        vals = window[sensor].values.astype(float)
        features[f"{sensor}_last"] = float(vals[-1])
        features[f"{sensor}_mean"] = float(np.mean(vals))
        features[f"{sensor}_std"] = float(np.std(vals, ddof=0))
        features[f"{sensor}_slope"] = float(np.dot(X_CENTERED, vals) / X_SUM_SQ)

    return features


def extract_causal_features_for_dataframe(
    df: pd.DataFrame,
    sensor_cols: list[str] = SENSOR_COLUMNS,
    window_size: int = WINDOW_SIZE,
    target_cap: int = 125,
    include_labels: bool = True,
    sample_weights_per_engine: bool = True,
) -> pd.DataFrame:
    """
    Extract causal rolling features for all eligible rows (cycle >= window_size) across all engines.
    
    - Never uses future readings or cross-engine data.
    - Yields feature table with unit_id, cycle, op settings, and sensor last/mean/std/slope.
    - If include_labels=True and max_cycle or failure endpoint is present, adds rul_true and rul_capped.
    - If sample_weights_per_engine=True, adds sample_weight such that sum(weight) == 1.0 per engine.
    """
    validate_trajectory_chronology(df, partition=str(df["source_partition"].iloc[0]) if "source_partition" in df.columns else "unknown")

    has_labels = include_labels and ("rul_true" in df.columns or "max_cycle" in df.columns)
    if include_labels and "rul_true" not in df.columns:
        # compute rul_true and rul_capped if max_cycle can be calculated
        max_cycles = df.groupby("unit_id")["cycle"].transform("max")
        df = df.copy()
        df["rul_true"] = max_cycles - df["cycle"]
        df["rul_capped"] = np.minimum(df["rul_true"], target_cap).astype(float)
        has_labels = True

    feature_rows = []
    
    # Process engine by engine
    for unit_id, group in df.groupby("unit_id", sort=True):
        group = group.sort_values("cycle").reset_index(drop=True)
        n_rows = len(group)
        if n_rows < window_size:
            continue
        
        # Precompute arrays for speed
        cycles = group["cycle"].values
        op1 = group["op_1"].values
        op2 = group["op_2"].values
        op3 = group["op_3"].values
        sensor_arrays = {s: group[s].values.astype(float) for s in sensor_cols}
        
        if has_labels:
            rul_true_arr = group["rul_true"].values
            rul_capped_arr = group["rul_capped"].values
            
        partition_val = group["source_partition"].iloc[0] if "source_partition" in group.columns else "train"

        engine_feature_rows = []
        for i in range(window_size - 1, n_rows):
            c_val = int(cycles[i])
            row_dict = {
                "unit_id": int(unit_id),
                "cycle": c_val,
                "source_partition": partition_val,
                "op_1": float(op1[i]),
                "op_2": float(op2[i]),
                "op_3": float(op3[i]),
            }
            
            for s in sensor_cols:
                w_vals = sensor_arrays[s][i - window_size + 1 : i + 1]
                row_dict[f"{s}_last"] = float(w_vals[-1])
                row_dict[f"{s}_mean"] = float(np.mean(w_vals))
                row_dict[f"{s}_std"] = float(np.std(w_vals, ddof=0))
                row_dict[f"{s}_slope"] = float(np.dot(X_CENTERED, w_vals) / X_SUM_SQ)
            
            if has_labels:
                row_dict["rul_true"] = float(rul_true_arr[i])
                row_dict["rul_capped"] = float(rul_capped_arr[i])
                
            engine_feature_rows.append(row_dict)
            
        # Add sample weight equal to 1.0 / count of eligible rows for this engine
        if sample_weights_per_engine and engine_feature_rows:
            w = 1.0 / len(engine_feature_rows)
            for r in engine_feature_rows:
                r["sample_weight"] = w
                
        feature_rows.extend(engine_feature_rows)

    result_df = pd.DataFrame(feature_rows)
    return result_df


def extract_snapshot_features(
    full_df: pd.DataFrame,
    snapshot_cutoffs: dict[str, dict],
    sensor_cols: list[str] = SENSOR_COLUMNS,
    window_size: int = WINDOW_SIZE,
    target_cap: int = 125,
) -> pd.DataFrame:
    """
    Extract single endpoint features for each engine at its assigned snapshot cut_cycle.
    """
    records = []
    for unit_id_str, meta in snapshot_cutoffs.items():
        unit_id = int(unit_id_str)
        cut_cycle = int(meta["cut_cycle"])
        engine_data = full_df[full_df["unit_id"] == unit_id]
        
        feats = extract_engine_features_at_cycle(
            engine_data,
            target_cycle=cut_cycle,
            sensor_cols=sensor_cols,
            window_size=window_size,
        )
        feats["unit_id"] = unit_id
        feats["source_partition"] = engine_data["source_partition"].iloc[0] if "source_partition" in engine_data.columns else "val"
        feats["lifetime_T"] = int(meta["lifetime_T"])
        feats["cut_cycle"] = cut_cycle
        feats["rul_true"] = float(meta["true_rul_at_cut"])
        feats["rul_capped"] = float(min(meta["true_rul_at_cut"], target_cap))
        feats["sample_weight"] = 1.0  # single snapshot per engine
        records.append(feats)

    return pd.DataFrame(records)
