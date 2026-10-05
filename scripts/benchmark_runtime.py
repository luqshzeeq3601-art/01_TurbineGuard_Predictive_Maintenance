"""Runtime and reproducibility benchmark script for TurbineGuard."""

import json
import os
import platform
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi.testclient import TestClient

from api.main import app
from turbineguard.artifacts import load_model_bundle
from turbineguard.config import load_config
from turbineguard.evaluate import compute_rul_metrics
from turbineguard.train import get_feature_columns_for_experiment


def measure_champion_retraining_reproducibility(cfg) -> dict:
    """
    Perform two separate, isolated training runs of the champion configuration (E05)
    on the real development dataset and compute validation endpoint RMSE for both runs.
    """
    dev_path = Path(cfg.data.features_dev)
    val_path = Path(cfg.data.features_val)

    if not (dev_path.exists() and val_path.exists()):
        return {
            "status": "skipped_missing_data",
            "message": "Processed features not available.",
        }

    dev_df = pd.read_parquet(dev_path)
    val_df = pd.read_parquet(val_path)

    const_sensors = ["s01", "s05", "s10", "s16", "s18", "s19"]
    feat_cols = get_feature_columns_for_experiment("E05", dev_df.columns.tolist(), const_sensors)

    champion_params = {
        "max_depth": 4,
        "n_estimators": 300,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": cfg.project.seed,
        "n_jobs": 2,
        "tree_method": "hist",
    }

    X_train = dev_df[feat_cols]
    y_train = dev_df["rul_capped"]
    w_train = dev_df["sample_weight"].values

    X_val = val_df[feat_cols]
    y_val_true = val_df["rul_true"].values

    # Run 1
    t0_run1 = time.perf_counter()
    model1 = xgb.XGBRegressor(**champion_params)
    model1.fit(X_train, y_train, sample_weight=w_train)
    t1_run1 = time.perf_counter()
    preds1 = np.maximum(0.0, model1.predict(X_val))
    m1 = compute_rul_metrics(preds1, y_val_true)

    # Run 2 (completely new model instance)
    t0_run2 = time.perf_counter()
    model2 = xgb.XGBRegressor(**champion_params)
    model2.fit(X_train, y_train, sample_weight=w_train)
    t1_run2 = time.perf_counter()
    preds2 = np.maximum(0.0, model2.predict(X_val))
    m2 = compute_rul_metrics(preds2, y_val_true)

    rmse_diff = abs(m1.rmse - m2.rmse)
    max_pred_diff = float(np.max(np.abs(preds1 - preds2)))
    is_reproducible = bool(rmse_diff <= 0.01)

    return {
        "champion_experiment_id": "E05",
        "model_family": "xgboost",
        "run1": {
            "validation_rmse": round(m1.rmse, 6),
            "validation_mae": round(m1.mae, 6),
            "train_duration_sec": round(t1_run1 - t0_run1, 3),
        },
        "run2": {
            "validation_rmse": round(m2.rmse, 6),
            "validation_mae": round(m2.mae, 6),
            "train_duration_sec": round(t1_run2 - t0_run2, 3),
        },
        "rmse_absolute_difference": round(rmse_diff, 8),
        "max_prediction_difference_cycles": round(max_pred_diff, 8),
        "reproducibility_target_le_0_01_cycles_achieved": is_reproducible,
    }


def run_runtime_benchmark(
    bundle_path: str = "models/v0.1.0",
    output_path: str = "reports/runtime.json",
) -> dict:
    """Measure startup, inference latency distribution, retraining reproducibility, and environment."""
    print("Measuring system environment and resources...")
    cfg = load_config("configs/default.yaml")

    sys_info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

    # 1. Cold start bundle loading
    t0 = time.perf_counter()
    bundle = load_model_bundle(bundle_path)
    cold_start_ms = (time.perf_counter() - t0) * 1000.0

    # Create a 20-cycle PRD single-engine history payload
    history_20 = []
    for c in range(1, 21):
        row = {"cycle": c, "op_1": -0.0005, "op_2": 0.0002, "op_3": 100.0}
        for s in range(1, 22):
            row[f"s{s:02d}"] = float(10.0 * s + 0.1 * c)
        history_20.append(row)

    payload_plain = {
        "dataset_id": "FD001",
        "engine_id": "unit_benchmark",
        "history": history_20,
        "explain": False,
    }
    payload_explain = {
        "dataset_id": "FD001",
        "engine_id": "unit_benchmark",
        "history": history_20,
        "explain": True,
    }

    # 2. Warm up client
    print("Warming up FastAPI client and executing latency benchmarks...")
    with TestClient(app) as client:
        # Warm-up call
        client.post("/predict", json=payload_plain)

        # 3. 100 Sequential scoring calls (warm)
        scoring_latencies_ms = []
        for _ in range(100):
            t_start = time.perf_counter()
            resp = client.post("/predict", json=payload_plain)
            t_end = time.perf_counter()
            assert resp.status_code == 200
            scoring_latencies_ms.append((t_end - t_start) * 1000.0)

        # 4. 20 Sequential scoring + explanation calls
        explain_latencies_ms = []
        for _ in range(20):
            t_start = time.perf_counter()
            resp = client.post("/predict", json=payload_explain)
            t_end = time.perf_counter()
            assert resp.status_code == 200
            explain_latencies_ms.append((t_end - t_start) * 1000.0)

    score_arr = np.array(scoring_latencies_ms)
    exp_arr = np.array(explain_latencies_ms)

    # 5. Measure Champion Retraining Reproducibility across 2 isolated runs
    print("Measuring champion retraining reproducibility across two isolated runs...")
    repro_data = measure_champion_retraining_reproducibility(cfg)

    report_data = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle_version": bundle.version,
        "environment": sys_info,
        "cold_start": {
            "bundle_load_ms": round(cold_start_ms, 2),
        },
        "scoring_latency_ms_n100": {
            "mean": round(float(np.mean(score_arr)), 2),
            "median_p50": round(float(np.percentile(score_arr, 50)), 2),
            "p90": round(float(np.percentile(score_arr, 90)), 2),
            "p95": round(float(np.percentile(score_arr, 95)), 2),
            "p99": round(float(np.percentile(score_arr, 99)), 2),
            "min": round(float(np.min(score_arr)), 2),
            "max": round(float(np.max(score_arr)), 2),
            "target_p95_under_200ms": bool(np.percentile(score_arr, 95) < 200.0),
        },
        "scoring_with_shap_latency_ms_n20": {
            "mean": round(float(np.mean(exp_arr)), 2),
            "median_p50": round(float(np.percentile(exp_arr, 50)), 2),
            "p95": round(float(np.percentile(exp_arr, 95)), 2),
        },
        "champion_retraining_reproducibility": repro_data,
        "container_evidence": {
            "dockerfile_present": Path("Dockerfile").exists(),
            "dockerignore_present": Path(".dockerignore").exists(),
            "docker_runtime_status": "unverified_host_daemon_stopped",
            "notes": "Dockerfile created with non-root appuser and read-only bundle mount; Docker Desktop daemon was not running on the local Windows host during verification.",
        },
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"Saved runtime and reproducibility benchmark report to {out_file}")
    return report_data


if __name__ == "__main__":
    run_runtime_benchmark()
