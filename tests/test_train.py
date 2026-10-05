"""Unit tests for baseline training, grouped cross-validation, and reproducibility."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbineguard.config import Config
from turbineguard.train import (
    get_feature_columns_for_experiment,
    train_and_evaluate_baseline_cv,
)


def create_synthetic_dev_features() -> tuple[pd.DataFrame, dict]:
    """Helper to create small synthetic development feature table and split manifest."""
    records = []
    # 10 engines total, 2 folds of 5 engines
    for u in range(1, 11):
        for c in range(20, 50):
            row = {
                "unit_id": u,
                "cycle": float(c),
                "op_1": 0.0,
                "op_2": 0.0,
                "op_3": 100.0,
                "s01_last": 10.0,
                "s02_last": float(640.0 + c * 0.1),
                "rul_true": float(50 - c),
                "rul_capped": float(min(50 - c, 125)),
                "sample_weight": 1.0 / 30.0,
            }
            records.append(row)
    
    df = pd.DataFrame(records)
    split_manifest = {
        "dataset_id": "FD001",
        "dev_engines": list(range(1, 11)),
        "cv_folds": {
            "fold_0": [1, 2, 3, 4, 5],
            "fold_1": [6, 7, 8, 9, 10],
        },
    }
    return df, split_manifest


def test_get_feature_columns_for_experiment():
    """Test feature column selection rules."""
    cols = ["cycle", "op_1", "op_2", "op_3", "s01_last", "s02_last", "s02_mean", "s02_std", "s02_slope"]
    const_sensors = ["s01"]
    
    e01_cols = get_feature_columns_for_experiment("E01", cols, const_sensors)
    assert e01_cols == ["cycle"]
    
    e02_cols = get_feature_columns_for_experiment("E02", cols, const_sensors)
    assert e02_cols == ["cycle"]
    
    e03_cols = get_feature_columns_for_experiment("E03", cols, const_sensors)
    assert "cycle" in e03_cols
    assert "s02_last" in e03_cols
    assert "s01_last" not in e03_cols  # Constant sensor excluded
    assert "s02_mean" not in e03_cols  # Rolling stats excluded in E03


def test_baseline_cv_runs_and_reproducible():
    """Test running baseline CV and check identical results on repeated runs."""
    df, split_manifest = create_synthetic_dev_features()
    cfg = Config()
    
    res1 = train_and_evaluate_baseline_cv("E02", df, split_manifest, cfg)
    res2 = train_and_evaluate_baseline_cv("E02", df, split_manifest, cfg)
    
    assert len(res1.fold_results) == 2
    assert np.isclose(res1.mean_cv_rmse, res2.mean_cv_rmse, atol=1e-6)
    assert np.isclose(res1.fold_results[0].rmse, res2.fold_results[0].rmse, atol=1e-6)


def test_baseline_real_data_reproducibility():
    """Test that training baselines on FD001 data produces identical metrics across two runs within 0.01 tolerance."""
    dev_path = Path("data/processed/features_dev.parquet")
    split_path = Path("data/processed/split_manifest.json")
    if not (dev_path.exists() and split_path.exists()):
        pytest.skip("Processed data not available.")
        
    dev_df = pd.read_parquet(dev_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)
    cfg = Config()
    
    # Run E03 twice
    run1 = train_and_evaluate_baseline_cv("E03", dev_df, split_manifest, cfg)
    run2 = train_and_evaluate_baseline_cv("E03", dev_df, split_manifest, cfg)
    
    assert abs(run1.mean_cv_rmse - run2.mean_cv_rmse) < 0.01
    for r1, r2 in zip(run1.fold_results, run2.fold_results, strict=False):
        assert abs(r1.rmse - r2.rmse) < 0.01
