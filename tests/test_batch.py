"""Parity and batch scoring tests between CLI and API."""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from turbineguard.predict import score_batch_history


def test_cli_api_prediction_parity(loaded_bundle, synthetic_history, portable_bundle_dir, monkeypatch):
    """Verify that CLI batch scoring and API endpoint return identical predictions."""
    bundle = loaded_bundle
    test_raw = synthetic_history
    monkeypatch.setenv("TURBINEGUARD_BUNDLE_DIR", str(portable_bundle_dir))
    
    # Select first 5 engines
    sample_df = test_raw[test_raw["unit_id"].isin([1, 2, 3, 4, 5])].copy()

    # 1. Local CLI scoring function
    cli_worklist, _ = score_batch_history(bundle, sample_df)

    # 2. REST API scoring
    with TestClient(app) as client:
        payload = {
            "readings": sample_df.to_dict(orient="records"),
            "explain": False,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        api_data = resp.json()["ranked_worklist"]

    api_df = pd.DataFrame(api_data)

    # Compare sort order and ranks
    assert list(cli_worklist["unit_id"]) == list(api_df["unit_id"])
    assert list(cli_worklist["rank"]) == list(api_df["rank"])

    # Compare numeric RUL values (within 1e-6 precision)
    np.testing.assert_allclose(
        cli_worklist["estimated_rul"].values,
        api_df["estimated_rul"].values,
        atol=1e-6,
        err_msg="Estimated RUL differs between CLI and API scoring",
    )

    # Compare anomaly scores (within 1e-6 precision)
    np.testing.assert_allclose(
        cli_worklist["anomaly_score"].values,
        api_df["anomaly_score"].values,
        atol=1e-6,
        err_msg="Anomaly score differs between CLI and API scoring",
    )

    # Compare boolean flags and priority labels
    assert list(cli_worklist["within_horizon"]) == list(api_df["within_horizon"])
    assert list(cli_worklist["anomaly_flag"]) == list(api_df["anomaly_flag"])
    assert list(cli_worklist["priority"]) == list(api_df["priority"])
