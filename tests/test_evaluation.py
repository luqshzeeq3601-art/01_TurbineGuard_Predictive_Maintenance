"""Unit tests for validation evaluation, bootstrap intervals, and gate verdicts."""

import numpy as np

from turbineguard.evaluate import compute_paired_bootstrap_confidence_intervals


def test_paired_bootstrap_confidence_intervals():
    """Verify bootstrap CI generates valid ordered confidence bounds."""
    rng = np.random.default_rng(42)
    y_true = np.array([10.0, 30.0, 60.0, 90.0] * 5)
    y_champ = y_true + rng.normal(0, 5, size=len(y_true))
    y_e01 = np.full_like(y_true, 88.0)
    y_e02 = y_true + rng.normal(0, 15, size=len(y_true))
    
    ci = compute_paired_bootstrap_confidence_intervals(
        y_champ, y_e01, y_e02, y_true, n_bootstraps=500, seed=42
    )
    
    # Check interval ordering: lower <= upper
    assert ci["rmse_95_ci"][0] <= ci["rmse_95_ci"][1]
    assert ci["mae_95_ci"][0] <= ci["mae_95_ci"][1]
    assert ci["diff_vs_e01_rmse_95_ci"][0] <= ci["diff_vs_e01_rmse_95_ci"][1]
    assert ci["diff_vs_e02_rmse_95_ci"][0] <= ci["diff_vs_e02_rmse_95_ci"][1]
    
    # Champion is much better than E01 -> difference is negative
    assert ci["diff_vs_e01_rmse_95_ci"][1] < 0.0
