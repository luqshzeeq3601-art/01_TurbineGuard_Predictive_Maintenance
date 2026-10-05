"""Label computation and anomaly proxy definitions for TurbineGuard."""

import numpy as np
import pandas as pd


def compute_rul_labels(
    df: pd.DataFrame,
    target_cap: int = 125,
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
) -> pd.DataFrame:
    """
    Compute true RUL and capped RUL labels for run-to-failure training data.
    
    For an engine with max cycle T at row cycle t:
    rul_true = T - t
    rul_capped = min(rul_true, target_cap)
    
    Terminal cycle has rul_true = 0.
    """
    df = df.copy()
    max_cycles = df.groupby(unit_col)[cycle_col].transform("max")
    df["max_cycle"] = max_cycles
    df["rul_true"] = df["max_cycle"] - df[cycle_col]
    df["rul_capped"] = np.minimum(df["rul_true"], target_cap).astype(float)
    return df


def assign_anomaly_proxies(
    df: pd.DataFrame,
    high_threshold: int = 100,
    near_failure_threshold: int = 30,
) -> pd.DataFrame:
    """
    Assign anomaly proxy categories based on true RUL:
    - proxy_high_rul: rul_true >= high_threshold (e.g. >= 100)
    - proxy_near_failure: rul_true <= near_failure_threshold (e.g. <= 30)
    - proxy_intermediate: not high and not near failure
    """
    df = df.copy()
    if "rul_true" not in df.columns:
        raise ValueError("Cannot assign anomaly proxies without 'rul_true' column.")

    df["proxy_high_rul"] = df["rul_true"] >= high_threshold
    df["proxy_near_failure"] = df["rul_true"] <= near_failure_threshold
    df["proxy_intermediate"] = (~df["proxy_high_rul"]) & (~df["proxy_near_failure"])
    return df
