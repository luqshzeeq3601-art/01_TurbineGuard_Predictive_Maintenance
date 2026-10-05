"""Evaluation metrics, engine bootstrap uncertainty, validation gates, and reports."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from turbineguard.anomaly import score_anomaly_samples
from turbineguard.config import load_config
from turbineguard.data import load_raw_cmapss_file
from turbineguard.features import extract_causal_features_for_dataframe
from turbineguard.labels import assign_anomaly_proxies, compute_rul_labels
from turbineguard.policy import (
    compute_capacity_diagnostics,
    compute_inspection_priority,
    is_within_horizon,
    rank_worklist,
)

logger = logging.getLogger(__name__)


@dataclass
class RULMetrics:
    """Comprehensive RUL regression metrics."""
    n_samples: int
    mae: float
    rmse: float
    bias: float
    overestimate_rate: float
    overestimate_gt_10_rate: float
    nasa_score_total: float
    nasa_score_mean: float
    band_metrics: dict[str, dict]


def compute_nasa_asymmetric_score(errors: np.ndarray) -> np.ndarray:
    """
    Compute NASA asymmetric scoring function for RUL error e = y_pred - y_true:
    - If e < 0 (early prediction / underestimate): exp(-e / 13) - 1
    - If e >= 0 (late prediction / overestimate): exp(e / 10) - 1
    """
    errors = np.asarray(errors, dtype=float)
    scores = np.empty_like(errors)
    
    under_mask = errors < 0
    scores[under_mask] = np.exp(-errors[under_mask] / 13.0) - 1.0
    scores[~under_mask] = np.exp(errors[~under_mask] / 10.0) - 1.0
    
    return scores


def compute_rul_metrics(
    y_pred: np.ndarray | list[float],
    y_true: np.ndarray | list[float],
    clip_zero: bool = True,
) -> RULMetrics:
    """
    Compute full evaluation metrics for RUL predictions against true uncapped RUL.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    
    if len(y_pred) != len(y_true):
        raise ValueError(f"Shape mismatch: y_pred ({len(y_pred)}) vs y_true ({len(y_true)})")
    if len(y_pred) == 0:
        raise ValueError("Cannot compute metrics on empty predictions.")

    if clip_zero:
        y_pred = np.maximum(0.0, y_pred)

    errors = y_pred - y_true
    n = len(errors)
    
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))
    over_rate = float(np.mean(errors > 0.0))
    over_10_rate = float(np.mean(errors > 10.0))
    
    nasa_scores = compute_nasa_asymmetric_score(errors)
    nasa_total = float(np.sum(nasa_scores))
    nasa_mean = float(np.mean(nasa_scores))

    bands = {
        "rul_le_30": y_true <= 30,
        "rul_31_to_100": (y_true > 30) & (y_true <= 100),
        "rul_gt_100": y_true > 100,
    }
    
    band_metrics = {}
    for band_name, mask in bands.items():
        count = int(np.sum(mask))
        if count > 0:
            b_err = errors[mask]
            b_nasa = nasa_scores[mask]
            band_metrics[band_name] = {
                "n_samples": count,
                "mae": float(np.mean(np.abs(b_err))),
                "rmse": float(np.sqrt(np.mean(b_err**2))),
                "bias": float(np.mean(b_err)),
                "nasa_score_mean": float(np.mean(b_nasa)),
            }
        else:
            band_metrics[band_name] = {
                "n_samples": 0,
                "mae": None,
                "rmse": None,
                "bias": None,
                "nasa_score_mean": None,
            }

    return RULMetrics(
        n_samples=n,
        mae=mae,
        rmse=rmse,
        bias=bias,
        overestimate_rate=over_rate,
        overestimate_gt_10_rate=over_10_rate,
        nasa_score_total=nasa_total,
        nasa_score_mean=nasa_mean,
        band_metrics=band_metrics,
    )


def compute_paired_bootstrap_confidence_intervals(
    y_pred_champ: np.ndarray,
    y_pred_e01: np.ndarray,
    y_pred_e02: np.ndarray,
    y_true: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compute 1,000 paired bootstrap draws across engines to estimate 95% confidence intervals.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    
    boot_rmse_champ = []
    boot_mae_champ = []
    boot_bias_champ = []
    boot_diff_e01 = []  # RMSE_champ - RMSE_e01 (negative means champion is better)
    boot_diff_e02 = []  # RMSE_champ - RMSE_e02

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        t_b = y_true[idx]
        c_b = y_pred_champ[idx]
        e01_b = y_pred_e01[idx]
        e02_b = y_pred_e02[idx]

        rmse_c = np.sqrt(np.mean((c_b - t_b)**2))
        rmse_01 = np.sqrt(np.mean((e01_b - t_b)**2))
        rmse_02 = np.sqrt(np.mean((e02_b - t_b)**2))

        boot_rmse_champ.append(rmse_c)
        boot_mae_champ.append(np.mean(np.abs(c_b - t_b)))
        boot_bias_champ.append(np.mean(c_b - t_b))
        boot_diff_e01.append(rmse_c - rmse_01)
        boot_diff_e02.append(rmse_c - rmse_02)

    return {
        "rmse_95_ci": [float(np.percentile(boot_rmse_champ, 2.5)), float(np.percentile(boot_rmse_champ, 97.5))],
        "mae_95_ci": [float(np.percentile(boot_mae_champ, 2.5)), float(np.percentile(boot_mae_champ, 97.5))],
        "bias_95_ci": [float(np.percentile(boot_bias_champ, 2.5)), float(np.percentile(boot_bias_champ, 97.5))],
        "diff_vs_e01_rmse_95_ci": [float(np.percentile(boot_diff_e01, 2.5)), float(np.percentile(boot_diff_e01, 97.5))],
        "diff_vs_e02_rmse_95_ci": [float(np.percentile(boot_diff_e02, 2.5)), float(np.percentile(boot_diff_e02, 97.5))],
    }


def run_validation_evaluation(config_path: str | Path = "configs/default.yaml") -> dict:
    """
    Execute full validation gate checks G3 and G4 on the frozen validation snapshot dataset
    and full validation trajectory anomaly diagnostics.
    """
    cfg = load_config(config_path)
    val_path = Path(cfg.data.features_val)
    staging_dir = Path(cfg.models.staging_dir)

    rul_model_path = staging_dir / "rul_pipeline.joblib"
    champ_info_path = staging_dir / "champion_info.json"
    anomaly_model_path = staging_dir / "anomaly_pipeline.joblib"
    anomaly_meta_path = staging_dir / "anomaly_metadata.json"

    if not (val_path.exists() and rul_model_path.exists() and anomaly_model_path.exists()):
        raise FileNotFoundError("Missing validation features or staged models. Run training and anomaly first.")

    val_df = pd.read_parquet(val_path)
    champ_info = json.loads(champ_info_path.read_text(encoding="utf-8"))
    anomaly_meta = json.loads(anomaly_meta_path.read_text(encoding="utf-8"))

    from turbineguard.artifacts import compute_file_sha256
    evaluated_model_hashes = {
        "rul_pipeline_sha256": compute_file_sha256(rul_model_path),
        "anomaly_pipeline_sha256": compute_file_sha256(anomaly_model_path),
    }

    rul_pipeline = joblib.load(rul_model_path)
    anomaly_pipeline = joblib.load(anomaly_model_path)

    # 1. RUL Predictions on Validation Snapshots
    feat_names = champ_info["feature_names"]
    preds_raw = rul_pipeline.predict(val_df[feat_names])
    preds_clipped = np.maximum(0.0, preds_raw)
    y_true = val_df["rul_true"].values

    metrics_champ = compute_rul_metrics(preds_clipped, y_true, clip_zero=True)

    # Baseline fits on development set
    dev_path = Path(cfg.data.features_dev)
    dev_df = pd.read_parquet(dev_path)
    
    m_e01 = DummyRegressor(strategy="median").fit(dev_df[["cycle"]], dev_df["rul_capped"], sample_weight=dev_df["sample_weight"])
    e01_preds = np.maximum(0.0, m_e01.predict(val_df[["cycle"]]))
    
    m_e02 = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=0.1, random_state=42))])
    m_e02.fit(dev_df[["cycle"]], dev_df["rul_capped"], reg__sample_weight=dev_df["sample_weight"])
    e02_preds = np.maximum(0.0, m_e02.predict(val_df[["cycle"]]))

    metrics_e01 = compute_rul_metrics(e01_preds, y_true, clip_zero=True)
    metrics_e02 = compute_rul_metrics(e02_preds, y_true, clip_zero=True)

    ci_dict = compute_paired_bootstrap_confidence_intervals(
        preds_clipped, e01_preds, e02_preds, y_true, n_bootstraps=1000, seed=cfg.project.seed
    )

    # 2. Gate G3 evaluation
    e01_rmse = metrics_e01.rmse
    e02_rmse = metrics_e02.rmse
    champ_rmse = metrics_champ.rmse

    improvement_vs_e01_pct = float((e01_rmse - champ_rmse) / e01_rmse * 100.0)
    improvement_vs_e02_pct = float((e02_rmse - champ_rmse) / e02_rmse * 100.0)
    is_nonconstant = champ_info["champion_experiment_id"] != "E01"

    gate_g3_pass = bool(is_nonconstant and improvement_vs_e01_pct >= 10.0 and improvement_vs_e02_pct >= 10.0)

    # 3. Anomaly evaluation on validation snapshots
    ano_feats = anomaly_meta["feature_names"]
    cutoff = anomaly_meta["frozen_cutoff_99th"]
    ano_scores_snap, ano_flags_snap = score_anomaly_samples(anomaly_pipeline, val_df, ano_feats, cutoff)

    val_df["estimated_rul"] = preds_clipped
    val_df["anomaly_score"] = ano_scores_snap
    val_df["anomaly_flag"] = ano_flags_snap
    val_df["within_horizon"] = val_df["estimated_rul"].apply(lambda r: is_within_horizon(r, cfg.policy.horizon_cycles))
    val_df["priority"] = val_df.apply(
        lambda row: compute_inspection_priority(row["within_horizon"], row["anomaly_flag"]), axis=1
    )

    # 4. Anomaly evaluation on full validation trajectories (Gate G4 protocol)
    split_manifest = json.loads(Path(cfg.data.split_manifest).read_text(encoding="utf-8"))
    val_engines = split_manifest["val_engines"]
    raw_train = load_raw_cmapss_file(Path(cfg.data.raw_dir) / "train_FD001.txt", "train")
    val_raw = raw_train[raw_train["unit_id"].isin(val_engines)].copy()
    val_labeled = compute_rul_labels(val_raw, target_cap=cfg.features.target_cap)
    val_full_feats = extract_causal_features_for_dataframe(val_labeled, include_labels=True, sample_weights_per_engine=False)
    val_full_feats = assign_anomaly_proxies(
        val_full_feats,
        high_threshold=cfg.features.high_rul_proxy_threshold,
        near_failure_threshold=cfg.features.near_failure_proxy_threshold,
    )

    val_full_scores, val_full_flags = score_anomaly_samples(anomaly_pipeline, val_full_feats, ano_feats, cutoff)
    val_full_feats["anomaly_score"] = val_full_scores
    val_full_feats["anomaly_flag"] = val_full_flags

    high_proxy_rows = val_full_feats[val_full_feats["proxy_high_rul"]]
    val_high_proxy_engines = int(high_proxy_rows["unit_id"].nunique())
    per_engine_false_flags = high_proxy_rows.groupby("unit_id")["anomaly_flag"].mean()
    macro_high_flag_rate = float(per_engine_false_flags.mean()) if val_high_proxy_engines > 0 else 0.0

    near_proxy_rows = val_full_feats[val_full_feats["proxy_near_failure"]]
    val_near_proxy_engines = int(near_proxy_rows["unit_id"].nunique())
    per_engine_near_flags = near_proxy_rows.groupby("unit_id")["anomaly_flag"].mean()
    macro_near_flag_rate = float(per_engine_near_flags.mean()) if val_near_proxy_engines > 0 else 0.0

    gate_g4_status = "PASS"
    if anomaly_meta["n_contributing_engines"] < 30 or val_high_proxy_engines < 10:
        gate_g4_status = "INCONCLUSIVE"
    elif macro_high_flag_rate > 0.05:
        gate_g4_status = "FAIL"
    else:
        gate_g4_status = "PASS"

    # 5. Policy & Capacity Diagnostics
    ranked_val = rank_worklist(val_df, rul_col="estimated_rul", anomaly_col="anomaly_flag", unit_col="unit_id")
    capacity_diag = compute_capacity_diagnostics(
        ranked_val,
        horizon_cycles=cfg.policy.horizon_cycles,
        capacity_fraction=cfg.policy.capacity_fraction,
        true_rul_col="rul_true",
    )

    report_dict = {
        "dataset_id": "FD001",
        "evaluated_model_hashes": evaluated_model_hashes,
        "champion": {
            "experiment_id": champ_info["champion_experiment_id"],
            "model_family": champ_info["model_family"],
            "description": champ_info["description"],
            "best_params": champ_info["best_params"],
        },
        "validation_metrics": {
            "n_validation_engines": len(val_df),
            "champion_rmse": champ_rmse,
            "champion_mae": metrics_champ.mae,
            "champion_bias": metrics_champ.bias,
            "champion_nasa_score_mean": metrics_champ.nasa_score_mean,
            "baseline_e01_rmse": e01_rmse,
            "baseline_e02_rmse": e02_rmse,
            "improvement_vs_e01_pct": improvement_vs_e01_pct,
            "improvement_vs_e02_pct": improvement_vs_e02_pct,
            "bootstrap_uncertainty": ci_dict,
            "band_metrics": metrics_champ.band_metrics,
        },
        "anomaly_metrics": {
            "frozen_cutoff": cutoff,
            "training_fitting_engines": anomaly_meta["n_contributing_engines"],
            "val_high_proxy_engines": val_high_proxy_engines,
            "val_high_proxy_macro_false_flag_rate": macro_high_flag_rate,
            "val_near_failure_proxy_engines": val_near_proxy_engines,
            "val_near_failure_proxy_macro_flag_rate": macro_near_flag_rate,
        },
        "gates": {
            "G3_RUL_promotion": {
                "outcome": "PASS" if gate_g3_pass else "FAIL",
                "criteria": "Nonconstant model >=10% better RMSE than E01 and E02",
                "champion_rmse": champ_rmse,
                "improvement_vs_e01_pct": improvement_vs_e01_pct,
                "improvement_vs_e02_pct": improvement_vs_e02_pct,
            },
            "G4_Anomaly_promotion": {
                "outcome": gate_g4_status,
                "criteria": ">=30 fitting engines, >=10 val proxy engines, high-RUL macro false-flag rate <=5%",
                "fitting_engines": anomaly_meta["n_contributing_engines"],
                "val_high_proxy_engines": val_high_proxy_engines,
                "high_proxy_macro_false_flag_rate": macro_high_flag_rate,
            },
        },
        "policy_diagnostics": asdict(capacity_diag),
    }

    report_path = Path("reports/validation_metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    logger.info(f"Saved validation metrics report to {report_path}")

    return report_dict


def run_official_test_evaluation(
    config_path: str | Path = "configs/default.yaml",
    bundle_dir: str | Path = "models/v0.1.0",
    release_id: str = "v0.1.0",
    allow_heldout_evaluation: bool = False,
    allow_overwrite: bool = False,
) -> dict:
    """
    Execute frozen official evaluation on NASA C-MAPSS FD001 holdout test set.
    
    CRITICAL EVALUATION CONTRACT:
    1. Bundle must be frozen.
    2. Score test engines without labels and write/hash predictions first.
    3. Load official test labels and evaluate.
    4. Record all results honestly including any missed aspirations.
    5. Preserve historical version reports and predictions.
    """
    if not allow_heldout_evaluation:
        raise PermissionError("Official heldout evaluation requires --allow-heldout-evaluation flag.")

    cfg = load_config(config_path)
    bundle_path = Path(bundle_dir)
    from turbineguard.artifacts import compute_file_sha256, load_model_bundle
    from turbineguard.data import load_raw_rul_file
    from turbineguard.predict import load_and_validate_input_file, score_batch_history

    logger.info(f"Loading frozen model bundle from {bundle_path}...")
    bundle = load_model_bundle(bundle_path)

    test_raw_path = Path(cfg.data.raw_dir) / "test_FD001.txt"
    rul_raw_path = Path(cfg.data.raw_dir) / "RUL_FD001.txt"

    logger.info("Scoring official test engines (unlabeled)...")
    test_df = load_and_validate_input_file(test_raw_path, input_format="cmapss")
    ranked_worklist, _test_features = score_batch_history(bundle, test_df)

    # 1. Save and hash predictions BEFORE loading true labels
    preds_csv_path = Path("reports") / (f"official_test_predictions_{release_id}.csv" if release_id != "v0.1.0" else "official_test_predictions.csv")
    if preds_csv_path.exists() and not allow_overwrite and release_id == "v0.1.0":
        # Preserve original v0.1.0 file
        pass
    else:
        preds_csv_path.parent.mkdir(parents=True, exist_ok=True)
        ranked_worklist.to_csv(preds_csv_path, index=False)
    preds_hash = compute_file_sha256(preds_csv_path)
    logger.info(f"Saved and hashed official test predictions: {preds_hash} -> {preds_csv_path}")

    # 2. Load official test RUL labels (100 engines in ascending unit_id order 1..100)
    logger.info("Loading official test RUL vector...")
    rul_official = load_raw_rul_file(rul_raw_path)

    # Merge labels strictly matching unit_id
    label_df = pd.DataFrame({
        "unit_id": list(range(1, len(rul_official) + 1)),
        "rul_true": rul_official,
    })
    merged_eval = pd.merge(ranked_worklist, label_df, on="unit_id", how="left")

    y_pred = merged_eval["estimated_rul"].values
    y_true = merged_eval["rul_true"].values

    # 3. Compute metrics against uncapped true RUL
    metrics_holdout = compute_rul_metrics(y_pred, y_true, clip_zero=True)

    # 4. Policy Diagnostics on Official Test
    capacity_diag = compute_capacity_diagnostics(
        merged_eval,
        horizon_cycles=bundle.policy["horizon_cycles"],
        capacity_fraction=bundle.policy["capacity_fraction"],
        true_rul_col="rul_true",
    )

    # 5. Check against aspirations
    aspiration_rmse_le_25 = bool(metrics_holdout.rmse <= 25.0)

    report_dict = {
        "dataset_id": "FD001",
        "release_id": release_id,
        "bundle_version": bundle.version,
        "promotion_status": bundle.metadata.get("promotion_status", "unknown"),
        "predictions_sha256": preds_hash,
        "n_test_engines": len(merged_eval),
        "holdout_metrics": {
            "rmse": metrics_holdout.rmse,
            "mae": metrics_holdout.mae,
            "bias": metrics_holdout.bias,
            "overestimate_rate": metrics_holdout.overestimate_rate,
            "overestimate_gt_10_rate": metrics_holdout.overestimate_gt_10_rate,
            "nasa_score_total": metrics_holdout.nasa_score_total,
            "nasa_score_mean": metrics_holdout.nasa_score_mean,
            "band_metrics": metrics_holdout.band_metrics,
        },
        "policy_diagnostics": asdict(capacity_diag),
        "aspirations": {
            "target_rmse_le_25": {
                "achieved": aspiration_rmse_le_25,
                "target": 25.0,
                "actual_rmse": metrics_holdout.rmse,
            },
        },
        "evaluation_notes": (
            "Official NASA C-MAPSS FD001 holdout evaluation on 100 test engines. "
            "Evaluated strictly against uncapped endpoint RUL. Predictions were hashed before joining labels."
        ),
    }

    metrics_json_path = Path("reports") / (f"official_test_metrics_{release_id}.json" if release_id != "v0.1.0" else "official_test_metrics.json")
    if metrics_json_path.exists() and not allow_overwrite and release_id == "v0.1.0":
        # Keep v0.1.0 preserved
        pass
    else:
        metrics_json_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        logger.info(f"Saved official holdout metrics report to {metrics_json_path}")
    return report_dict
