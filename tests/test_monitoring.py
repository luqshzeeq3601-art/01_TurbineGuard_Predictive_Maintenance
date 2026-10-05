"""Tests for Evidently drift reporting, insufficient data handling, and drift controls."""

import json

import pandas as pd
import pytest

from turbineguard.artifacts import load_model_bundle
from turbineguard.monitoring import run_drift_analysis, run_drift_controls


@pytest.fixture
def loaded_bundle():
    return load_model_bundle("models/v0.1.0")


def test_monitoring_reference_exists(loaded_bundle):
    ref_path = loaded_bundle.bundle_dir / "monitoring_reference.parquet"
    assert ref_path.exists()
    df = pd.read_parquet(ref_path)
    assert len(df) == 80  # 80 development engine snapshots


def test_drift_analysis_insufficient_data(loaded_bundle):
    ref_path = loaded_bundle.bundle_dir / "monitoring_reference.parquet"
    df = pd.read_parquet(ref_path)
    
    # Pass only 15 rows (< 30 threshold)
    small_df = df.iloc[:15].copy()
    summary = run_drift_analysis(loaded_bundle, small_df, min_engines_required=30)

    assert summary.status == "insufficient_data"
    assert summary.dataset_drift is False
    assert summary.number_of_drifted_columns == 0
    assert "below statistical reliability threshold" in summary.details["message"]


def test_drift_controls_execution(loaded_bundle, tmp_path):
    controls_dir = tmp_path / "controls"
    res = run_drift_controls(loaded_bundle, output_dir=controls_dir)

    assert res["all_controls_passed"] is True
    assert res["controls"]["no_change_control"]["passed"] is True
    assert res["controls"]["shifted_sensor_control"]["passed"] is True
    assert res["controls"]["insufficient_data_control"]["passed"] is True

    # Verify control_results.json exists and is valid
    res_file = controls_dir / "control_results.json"
    assert res_file.exists()
    data = json.loads(res_file.read_text(encoding="utf-8"))
    assert data["all_controls_passed"] is True
