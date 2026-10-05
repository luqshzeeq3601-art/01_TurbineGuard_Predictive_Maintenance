"""Tests for official holdout evaluation outputs and hashing contract."""

import json
from pathlib import Path

import pytest

from turbineguard.artifacts import compute_file_sha256


@pytest.mark.integration
def test_official_test_predictions_and_metrics_integrity():
    preds_file = Path("reports/official_test_predictions.csv")
    metrics_file = Path("reports/official_test_metrics.json")

    if not (preds_file.exists() and metrics_file.exists()):
        pytest.skip("Official test evaluation reports not present locally.")

    metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))

    # Verify predictions hash matches record in metrics report
    computed_hash = compute_file_sha256(preds_file)
    assert computed_hash == metrics_data["predictions_sha256"]

    # Verify metrics structure
    assert metrics_data["n_test_engines"] == 100
    assert metrics_data["holdout_metrics"]["rmse"] > 0
    assert metrics_data["holdout_metrics"]["rmse"] <= 25.0
    assert metrics_data["aspirations"]["target_rmse_le_25"]["achieved"] is True


@pytest.mark.integration
def test_official_test_predictions_and_metrics_integrity_v0_2_0():
    preds_file = Path("reports/official_test_predictions_v0.2.0.csv")
    metrics_file = Path("reports/official_test_metrics_v0.2.0.json")

    if not (preds_file.exists() and metrics_file.exists()):
        pytest.skip("Official test evaluation reports for v0.2.0 not present locally.")

    metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))

    # Verify predictions hash matches record in metrics report
    computed_hash = compute_file_sha256(preds_file)
    assert computed_hash == metrics_data["predictions_sha256"]

    # Verify metrics structure
    assert metrics_data["n_test_engines"] == 100
    assert metrics_data["holdout_metrics"]["rmse"] > 0
    assert metrics_data["holdout_metrics"]["rmse"] <= 25.0
    assert metrics_data["aspirations"]["target_rmse_le_25"]["achieved"] is True
