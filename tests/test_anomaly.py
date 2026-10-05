"""Unit tests for Isolation Forest anomaly detection component."""

import numpy as np
import pandas as pd

from turbineguard.anomaly import (
    fit_isolation_forest_component,
    get_anomaly_sensor_feature_columns,
    sample_high_rul_proxy_subset,
    score_anomaly_samples,
)
from turbineguard.config import Config, ModelsConfig


def create_synthetic_proxy_dev_df(n_engines: int = 35) -> pd.DataFrame:
    """Helper to create synthetic dev feature table with realistic proxy rows."""
    rng = np.random.default_rng(42)
    records = []
    for u in range(1, n_engines + 1):
        base_s2 = 640.0 + rng.normal(0, 1.0)
        for c in range(20, 150):
            rul = 150 - c
            row = {
                "unit_id": u,
                "cycle": c,
                "rul_true": float(rul),
                "rul_capped": float(min(rul, 125)),
                "s01_last": 10.0,
                "s02_last": float(base_s2 + rng.normal(0, 0.2)),
                "s02_mean": float(base_s2),
                "s02_std": 0.2,
                "s02_slope": 0.0,
            }
            records.append(row)
    return pd.DataFrame(records)


def test_get_anomaly_sensor_feature_columns_excludes_non_sensors():
    """Verify anomaly features exclude cycle, unit_id, op settings, and RUL."""
    cols = ["unit_id", "cycle", "op_1", "op_2", "op_3", "rul_true", "rul_capped", "s01_last", "s02_last", "s02_mean", "s02_std", "s02_slope"]
    const_sensors = ["s01"]
    
    ano_cols = get_anomaly_sensor_feature_columns(cols, const_sensors)
    assert "s02_last" in ano_cols
    assert "s02_mean" in ano_cols
    assert "s01_last" not in ano_cols
    assert "cycle" not in ano_cols
    assert "unit_id" not in ano_cols
    assert "op_1" not in ano_cols
    assert "rul_true" not in ano_cols


def test_sample_high_rul_proxy_subset():
    """Test sampling logic (only rul_true >= 100, max 50 rows per engine)."""
    df = create_synthetic_proxy_dev_df(n_engines=35)
    sample = sample_high_rul_proxy_subset(df, high_threshold=100, max_rows_per_engine=50, min_required_engines=30)
    
    assert sample["unit_id"].nunique() == 35
    assert (sample["rul_true"] >= 100).all()
    # Check max rows per engine
    counts = sample.groupby("unit_id").size()
    assert (counts <= 50).all()


def test_fit_and_score_isolation_forest(tmp_path):
    """Test fitting Isolation Forest, score direction, and cutoff boundary."""
    df = create_synthetic_proxy_dev_df(n_engines=35)
    cfg = Config(models=ModelsConfig(staging_dir=str(tmp_path / "staging")))
    
    pipeline, fit_res = fit_isolation_forest_component(df, cfg)
    assert fit_res.n_contributing_engines == 35
    assert fit_res.frozen_cutoff_99th > 0.0
    
    # Inlier vs synthetic extreme outlier
    inlier = df[df["rul_true"] >= 100].head(5)
    outlier = inlier.copy()
    outlier["s02_last"] = 665.0  # +25 sigma shift
    outlier["s02_mean"] = 665.0
    
    inlier_scores, _ = score_anomaly_samples(pipeline, inlier, fit_res.feature_names, fit_res.frozen_cutoff_99th)
    outlier_scores, outlier_flags = score_anomaly_samples(pipeline, outlier, fit_res.feature_names, fit_res.frozen_cutoff_99th)
    
    # Outlier should have strictly higher anomaly score than normal inlier
    assert (outlier_scores > inlier_scores).all()
    assert outlier_flags.all()
