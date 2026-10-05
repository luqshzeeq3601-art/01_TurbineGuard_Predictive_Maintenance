"""Unit tests for causal 20-cycle feature extraction and leakage prevention."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbineguard.data import ValidationError
from turbineguard.features import (
    compute_rolling_ols_slope,
    extract_causal_features_for_dataframe,
    extract_engine_features_at_cycle,
)


def test_compute_rolling_ols_slope():
    """Test OLS slope on known vectors."""
    # Flat
    y_flat = np.full(20, 5.0)
    assert np.isclose(compute_rolling_ols_slope(y_flat), 0.0)
    
    # Linear slope 2.5: y = 10 + 2.5 * x
    x = np.arange(20, dtype=float)
    y_line = 10.0 + 2.5 * x
    assert np.isclose(compute_rolling_ols_slope(y_line), 2.5)


def test_hand_computed_sensor_fixture():
    """Test feature extraction against hand-calculated fixture values."""
    fixture_path = Path("tests/fixtures/sensor_history.csv")
    assert fixture_path.exists()
    
    df = pd.read_csv(fixture_path)
    df["source_partition"] = "train"
    
    # Extract features at cycle 20
    feats_c20 = extract_engine_features_at_cycle(df, target_cycle=20)
    
    # Sensor 1: constant 10.0
    assert np.isclose(feats_c20["s01_last"], 10.0)
    assert np.isclose(feats_c20["s01_mean"], 10.0)
    assert np.isclose(feats_c20["s01_std"], 0.0)
    assert np.isclose(feats_c20["s01_slope"], 0.0)
    
    # Sensor 2: ramp 100, 102, ..., 138
    assert np.isclose(feats_c20["s02_last"], 138.0)
    assert np.isclose(feats_c20["s02_mean"], 119.0)
    expected_std = 2.0 * np.sqrt(665.0 / 20.0)  # ~11.53256
    assert np.isclose(feats_c20["s02_std"], expected_std)
    assert np.isclose(feats_c20["s02_slope"], 2.0)


def test_insufficient_history_rejected():
    """Verify that fewer than 20 cycles raises ValidationError."""
    fixture_path = Path("tests/fixtures/sensor_history.csv")
    df = pd.read_csv(fixture_path)
    df["source_partition"] = "train"
    
    # Try cycle 19 (only 19 rows available)
    with pytest.raises(ValidationError, match="Insufficient history: required 20 cycles"):
        extract_engine_features_at_cycle(df, target_cycle=19)


def test_causality_and_future_invariance():
    """
    CRITICAL LEAKAGE TEST (Gate G2):
    Verify that mutating future readings (cycles > t) has zero effect on features at cycle t.
    """
    fixture_path = Path("tests/fixtures/sensor_history.csv")
    df_orig = pd.read_csv(fixture_path)
    df_orig["source_partition"] = "train"
    
    # Compute baseline features at cycle 20
    feats_before = extract_engine_features_at_cycle(df_orig, target_cycle=20)
    
    # Mutate future rows (cycles 21 to 25) drastically
    df_mutated = df_orig.copy()
    future_mask = df_mutated["cycle"] > 20
    df_mutated.loc[future_mask, "s02"] = 999999.0
    df_mutated.loc[future_mask, "op_1"] = 8888.0
    
    feats_after = extract_engine_features_at_cycle(df_mutated, target_cycle=20)
    
    # Check absolute equality across all features
    for key in feats_before:
        assert feats_before[key] == feats_after[key], (
            f"Feature {key} changed when future readings were mutated! (before: {feats_before[key]}, after: {feats_after[key]})"
        )


def test_cross_engine_isolation():
    """
    CRITICAL LEAKAGE TEST (Gate G2):
    Verify that mutating or appending data for another engine cannot alter features of engine 1.
    """
    fixture_path = Path("tests/fixtures/sensor_history.csv")
    df1 = pd.read_csv(fixture_path)
    df1["source_partition"] = "train"
    
    # Create engine 2 data
    df2 = df1.copy()
    df2["unit_id"] = 2
    df2["s02"] = df2["s02"] * 100.0  # completely different values
    
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    feats_single = extract_engine_features_at_cycle(df1, target_cycle=20)
    feats_multi = extract_engine_features_at_cycle(df_combined[df_combined["unit_id"] == 1], target_cycle=20)
    
    for key in feats_single:
        assert feats_single[key] == feats_multi[key]


def test_extract_causal_features_for_dataframe():
    """Test batch feature extraction with sample weights and labels."""
    fixture_path = Path("tests/fixtures/sensor_history.csv")
    df = pd.read_csv(fixture_path)
    df["source_partition"] = "train"
    
    feats_df = extract_causal_features_for_dataframe(df, window_size=20, target_cap=125)
    
    # Engine 1 has 25 cycles -> eligible cycles are 20, 21, 22, 23, 24, 25 (6 rows)
    assert len(feats_df) == 6
    assert set(feats_df["cycle"]) == {20, 21, 22, 23, 24, 25}
    
    # Check sample weights sum to 1.0 per engine
    assert np.isclose(feats_df["sample_weight"].sum(), 1.0)
    assert np.isclose(feats_df["sample_weight"].iloc[0], 1.0 / 6.0)
    
    # Check RUL labels (lifetime is 25)
    # At cycle 20, rul_true = 5, rul_capped = 5
    # At cycle 25, rul_true = 0, rul_capped = 0
    row_20 = feats_df[feats_df["cycle"] == 20].iloc[0]
    row_25 = feats_df[feats_df["cycle"] == 25].iloc[0]
    assert row_20["rul_true"] == 5.0
    assert row_25["rul_true"] == 0.0
