"""Maintenance inspection policy, priority assignments, and capacity diagnostics."""

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class CapacityDiagnostics:
    total_engines: int
    capacity_k: int
    actual_positives_count: int  # true RUL <= horizon
    actual_prevalence: float
    selected_positives_count: int  # true positives in top-k
    precision_at_k: float
    recall_at_k: float
    lift_over_prevalence: float


def is_within_horizon(estimated_rul_cycles: float, horizon_cycles: int = 30) -> bool:
    """Return True if estimated RUL is within horizon (inclusive <= horizon_cycles)."""
    return float(estimated_rul_cycles) <= float(horizon_cycles)


def compute_inspection_priority(within_horizon: bool, anomaly_flag: bool) -> str:
    """
    Assign inspection priority category:
    - review_soon: when estimated RUL is within horizon (<= 30 cycles)
    - investigate: when not within horizon, but anomaly flag is True
    - routine_review: otherwise
    """
    if within_horizon:
        return "review_soon"
    elif anomaly_flag:
        return "investigate"
    else:
        return "routine_review"


def rank_worklist(
    df: pd.DataFrame,
    rul_col: str = "estimated_rul",
    anomaly_col: str = "anomaly_flag",
    unit_col: str = "unit_id",
) -> pd.DataFrame:
    """
    Sort worklist deterministically:
    1. Ascending estimated RUL (most urgent first)
    2. Descending anomaly flag (anomalous first in tie)
    3. Ascending string unit identifier (deterministic tie-breaker)
    """
    df = df.copy()
    # Create temp string ID for sort stability
    df["_unit_str"] = df[unit_col].astype(str)
    
    # Sort
    ranked = df.sort_values(
        by=[rul_col, anomaly_col, "_unit_str"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    
    ranked.drop(columns=["_unit_str"], inplace=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def compute_capacity_diagnostics(
    ranked_df: pd.DataFrame,
    horizon_cycles: int = 30,
    capacity_fraction: float = 0.20,
    true_rul_col: str = "rul_true",
) -> CapacityDiagnostics:
    """
    Compute capacity diagnostics at top k = max(1, ceil(capacity_fraction * N)).
    """
    n = len(ranked_df)
    if n == 0:
        raise ValueError("Cannot compute capacity diagnostics on empty dataframe.")

    k = max(1, math.ceil(capacity_fraction * n))
    top_k = ranked_df.iloc[:k]

    if true_rul_col not in ranked_df.columns:
        raise ValueError(f"Column '{true_rul_col}' required to compute capacity diagnostics.")

    actual_positives = (ranked_df[true_rul_col] <= horizon_cycles).sum()
    prevalence = float(actual_positives / n)

    selected_positives = (top_k[true_rul_col] <= horizon_cycles).sum()
    precision_k = float(selected_positives / k)
    recall_k = float(selected_positives / actual_positives) if actual_positives > 0 else 0.0
    lift = float(precision_k / prevalence) if prevalence > 0 else 0.0

    return CapacityDiagnostics(
        total_engines=n,
        capacity_k=k,
        actual_positives_count=int(actual_positives),
        actual_prevalence=prevalence,
        selected_positives_count=int(selected_positives),
        precision_at_k=precision_k,
        recall_at_k=recall_k,
        lift_over_prevalence=lift,
    )
