"""Unit tests for inspection policy, priority assignments, and worklist ranking."""

import numpy as np
import pandas as pd

from turbineguard.policy import (
    compute_capacity_diagnostics,
    compute_inspection_priority,
    is_within_horizon,
    rank_worklist,
)


def test_is_within_horizon():
    """Verify inclusive horizon boundary (<= 30)."""
    assert is_within_horizon(30.0, 30) is True
    assert is_within_horizon(29.9, 30) is True
    assert is_within_horizon(0.0, 30) is True
    assert is_within_horizon(30.1, 30) is False
    assert is_within_horizon(45.0, 30) is False


def test_compute_inspection_priority():
    """Verify priority assignment rules."""
    # within_horizon=True -> review_soon (regardless of anomaly)
    assert compute_inspection_priority(within_horizon=True, anomaly_flag=False) == "review_soon"
    assert compute_inspection_priority(within_horizon=True, anomaly_flag=True) == "review_soon"
    
    # within_horizon=False, anomaly=True -> investigate
    assert compute_inspection_priority(within_horizon=False, anomaly_flag=True) == "investigate"
    
    # within_horizon=False, anomaly=False -> routine_review
    assert compute_inspection_priority(within_horizon=False, anomaly_flag=False) == "routine_review"


def test_rank_worklist():
    """Verify deterministic ranking: ascending RUL, descending anomaly_flag, ascending unit_id."""
    df = pd.DataFrame({
        "unit_id": [10, 2, 5, 1],
        "estimated_rul": [20.0, 20.0, 10.0, 50.0],
        "anomaly_flag": [False, True, False, False],
    })
    
    ranked = rank_worklist(df, rul_col="estimated_rul", anomaly_col="anomaly_flag", unit_col="unit_id")
    
    # Expected order:
    # 1. unit 5 (RUL 10.0)
    # 2. unit 2 (RUL 20.0, anomaly True)
    # 3. unit 10 (RUL 20.0, anomaly False)
    # 4. unit 1 (RUL 50.0)
    assert ranked["unit_id"].tolist() == [5, 2, 10, 1]


def test_compute_capacity_diagnostics():
    """Test capacity diagnostics with ceil rounding."""
    # 10 engines, capacity 20% -> k = ceil(0.20 * 10) = 2
    # 3 actual positives (RUL <= 30)
    df = pd.DataFrame({
        "unit_id": list(range(1, 11)),
        "estimated_rul": [10.0, 20.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "anomaly_flag": [False] * 10,
        "rul_true": [15.0, 25.0, 30.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
    })
    
    diag = compute_capacity_diagnostics(df, horizon_cycles=30, capacity_fraction=0.20, true_rul_col="rul_true")
    assert diag.total_engines == 10
    assert diag.capacity_k == 2
    assert diag.actual_positives_count == 3
    assert diag.selected_positives_count == 2
    assert diag.precision_at_k == 1.0  # 2/2
    assert np.isclose(diag.recall_at_k, 2.0 / 3.0)
