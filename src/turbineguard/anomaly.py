"""Isolation Forest anomaly detection component on high-RUL proxy baseline."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from turbineguard.config import Config
from turbineguard.data import SENSOR_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class AnomalyFitResult:
    n_contributing_engines: int
    n_sampled_rows: int
    feature_names: list[str]
    frozen_cutoff_99th: float
    min_score: float
    max_score: float
    mean_score: float
    pipeline_path: str
    pipeline_sha256: str = ""
    dev_engines: list[int] | None = None


def get_anomaly_sensor_feature_columns(df_columns: list[str], constant_sensors: list[str]) -> list[str]:
    """
    Get sensor-only feature columns (last, mean, std, slope) for nonconstant sensors.
    Strictly excludes cycle, unit_id, op settings, RUL targets, and flags.
    """
    sensor_cols = []
    for s in SENSOR_COLUMNS:
        if s not in constant_sensors:
            for suffix in ["last", "mean", "std", "slope"]:
                col = f"{s}_{suffix}"
                if col in df_columns:
                    sensor_cols.append(col)
                elif suffix == "last" and s in df_columns:
                    sensor_cols.append(s)
    return sensor_cols


def sample_high_rul_proxy_subset(
    dev_df: pd.DataFrame,
    high_threshold: int = 100,
    max_rows_per_engine: int = 50,
    min_required_engines: int = 30,
) -> pd.DataFrame:
    """
    Sample at most max_rows_per_engine evenly spaced rows per development engine
    from eligible rows where rul_true >= high_threshold.
    
    Requires at least min_required_engines.
    """
    if "rul_true" not in dev_df.columns:
        raise ValueError("rul_true column required to identify high-RUL proxy subset.")

    proxy_df = dev_df[dev_df["rul_true"] >= high_threshold].copy()
    contributing_engines = proxy_df["unit_id"].nunique()

    if contributing_engines < min_required_engines:
        raise ValueError(
            f"Insufficient support for anomaly proxy: found {contributing_engines} engines, "
            f"required at least {min_required_engines}."
        )

    sampled_records = []
    for unit_id, group in proxy_df.groupby("unit_id", sort=True):
        n = len(group)
        if n <= max_rows_per_engine:
            sampled_records.append(group)
        else:
            # Evenly spaced indices
            indices = np.round(np.linspace(0, n - 1, max_rows_per_engine)).astype(int)
            sampled_records.append(group.iloc[indices])

    result = pd.concat(sampled_records, ignore_index=True)
    return result


def fit_isolation_forest_component(
    dev_df: pd.DataFrame,
    cfg: Config,
) -> tuple[Pipeline, AnomalyFitResult]:
    """
    Fit Isolation Forest on high-RUL proxy subset and compute frozen 99th percentile cutoff.
    """
    constant_sensors = ["s01", "s05", "s10", "s16", "s18", "s19"]
    proxy_sample = sample_high_rul_proxy_subset(
        dev_df,
        high_threshold=cfg.features.high_rul_proxy_threshold,
        max_rows_per_engine=50,
        min_required_engines=30,
    )

    feat_cols = get_anomaly_sensor_feature_columns(dev_df.columns.tolist(), constant_sensors)
    X_proxy = proxy_sample[feat_cols].values

    # Pipeline: StandardScaler + IsolationForest
    iso = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=cfg.project.seed,
        n_jobs=2,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", iso),
    ])
    pipeline.fit(X_proxy)

    # Anomaly score = -score_samples (higher is more unusual)
    # Note: pipeline.score_samples is available by forwarding to model
    X_scaled = pipeline.named_steps["scaler"].transform(X_proxy)
    raw_scores = pipeline.named_steps["model"].score_samples(X_scaled)
    anomaly_scores = -raw_scores

    cutoff = float(np.percentile(anomaly_scores, cfg.policy.anomaly_percentile, method="linear"))

    staging_dir = Path(cfg.models.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    pipe_path = staging_dir / "anomaly_pipeline.joblib"
    joblib.dump(pipeline, pipe_path)

    import hashlib
    hasher = hashlib.sha256()
    with open(pipe_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    pipe_sha = hasher.hexdigest()

    contributing_uids = sorted(proxy_sample["unit_id"].unique().tolist())

    fit_result = AnomalyFitResult(
        n_contributing_engines=len(contributing_uids),
        n_sampled_rows=len(proxy_sample),
        feature_names=feat_cols,
        frozen_cutoff_99th=cutoff,
        min_score=float(np.min(anomaly_scores)),
        max_score=float(np.max(anomaly_scores)),
        mean_score=float(np.mean(anomaly_scores)),
        pipeline_path=str(pipe_path),
        pipeline_sha256=pipe_sha,
        dev_engines=contributing_uids,
    )

    metadata_path = staging_dir / "anomaly_metadata.json"
    metadata_path.write_text(json.dumps(asdict(fit_result), indent=2), encoding="utf-8")

    logger.info(
        f"Fitted Isolation Forest on {fit_result.n_contributing_engines} engines "
        f"({fit_result.n_sampled_rows} rows, {len(feat_cols)} features). "
        f"Frozen 99th percentile cutoff = {cutoff:.4f}, sha256={pipe_sha[:8]}..."
    )
    return pipeline, fit_result


def score_anomaly_samples(
    pipeline: Pipeline,
    df: pd.DataFrame,
    feature_names: list[str],
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score input samples:
    - anomaly_score = -score_samples(X) (higher is more unusual)
    - anomaly_flag = anomaly_score > cutoff
    """
    X = df[feature_names].values
    X_scaled = pipeline.named_steps["scaler"].transform(X)
    raw_scores = pipeline.named_steps["model"].score_samples(X_scaled)
    scores = -raw_scores
    flags = scores > cutoff
    return scores, flags
