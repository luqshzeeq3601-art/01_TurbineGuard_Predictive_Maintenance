"""Integration and comprehensive positive/negative contract tests for FastAPI inference endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app, bundle_state


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def make_valid_history(n_cycles: int = 25) -> list[dict]:
    """Helper to generate a valid synthetic single-engine sensor history."""
    history = []
    for c in range(1, n_cycles + 1):
        row = {
            "cycle": c,
            "op_1": -0.0005,
            "op_2": 0.0002,
            "op_3": 100.0,
        }
        for s in range(1, 22):
            row[f"s{s:02d}"] = float(10.0 * s + 0.1 * c)
        history.append(row)
    return history


def test_api_health(client):
    """Test health endpoint liveness probe."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp_utc" in data


def test_api_ready(client):
    """Test ready endpoint readiness probe."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert "bundle_version" in data


def test_api_model_info(client):
    """Test model-info metadata endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"] == "FD001"
    assert "policy" in data
    assert "feature_schema" in data
    assert "validation_gates" in data
    assert "file_hashes" in data


def test_api_single_engine_predict_positive(client):
    """Positive test: PRD single-engine prediction contract with explanations."""
    payload = {
        "dataset_id": "FD001",
        "engine_id": "turbine_unit_042",
        "history": make_valid_history(25),
        "explain": True,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["engine_id"] == "turbine_unit_042"
    assert data["dataset_id"] == "FD001"
    assert data["latest_cycle"] == 25
    assert data["history_length"] == 25
    assert isinstance(data["estimated_rul_cycles"], float)
    assert data["estimated_rul_cycles"] >= 0.0
    assert isinstance(data["within_horizon"], bool)
    assert isinstance(data["anomaly_score"], float)
    assert isinstance(data["anomaly_flag"], bool)
    assert data["inspection_priority"] in ["review_soon", "investigate", "routine_review"]
    assert "bundle_version" in data
    assert "policy_version" in data
    assert "schema_version" in data

    # Check explanation structure
    assert data["explanation"] is not None
    exp = data["explanation"]
    assert "raw_output" in exp
    assert "base_value" in exp
    assert "clipping_applied" in exp
    assert len(exp["top_contributions"]) == 3
    for contrib in exp["top_contributions"]:
        assert "feature" in contrib
        assert "feature_value" in contrib
        assert "contribution_cycles" in contrib
        assert "effect" in contrib


def test_api_predict_rejects_unknown_fields(client):
    """Negative test: reject unknown fields (extra='forbid')."""
    payload = {
        "dataset_id": "FD001",
        "engine_id": "unit_1",
        "history": make_valid_history(20),
        "unknown_extra_field": "disallowed",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_predict_rejects_insufficient_history(client):
    """Negative test: reject history with fewer than 20 cycles."""
    payload = {
        "dataset_id": "FD001",
        "engine_id": "unit_1",
        "history": make_valid_history(19),
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_predict_rejects_non_consecutive_cycles(client):
    """Negative test: reject history with cycle gaps."""
    hist = make_valid_history(20)
    hist[10]["cycle"] = 15  # gap: 1..10, 15, 12..
    payload = {
        "dataset_id": "FD001",
        "engine_id": "unit_1",
        "history": hist,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_predict_rejects_invalid_dataset_id(client):
    """Negative test: reject unsupported dataset_id."""
    payload = {
        "dataset_id": "FD002",
        "engine_id": "unit_1",
        "history": make_valid_history(20),
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_predict_rejects_oversized_payload(client):
    """Negative test: reject request body exceeding 1 MiB limit (HTTP 413)."""
    # Create large dummy history with 250 cycles or huge extra data
    huge_data = "x" * (1024 * 1024 + 100)
    response = client.post(
        "/predict",
        content=huge_data,
        headers={"Content-Length": str(len(huge_data)), "Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_api_ready_unavailable_bundle():
    """Negative test: /ready returns 503 when bundle is unavailable."""
    original_bundle = bundle_state["bundle"]
    try:
        bundle_state["bundle"] = None
        # Mock non-existent env
        import os
        old_env = os.environ.get("TURBINEGUARD_BUNDLE_DIR")
        os.environ["TURBINEGUARD_BUNDLE_DIR"] = "nonexistent_bundle_dir"

        with TestClient(app, raise_server_exceptions=False) as test_client:
            resp = test_client.get("/ready")
            assert resp.status_code == 503

            resp_info = test_client.get("/model-info")
            assert resp_info.status_code == 503
    finally:
        bundle_state["bundle"] = original_bundle
        if old_env is not None:
            os.environ["TURBINEGUARD_BUNDLE_DIR"] = old_env
        elif "TURBINEGUARD_BUNDLE_DIR" in os.environ:
            del os.environ["TURBINEGUARD_BUNDLE_DIR"]
