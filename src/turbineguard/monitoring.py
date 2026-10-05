"""Data drift reporting, Evidently integration, and controlled drift checks."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

from turbineguard.artifacts import LoadedBundle

logger = logging.getLogger(__name__)


@dataclass
class DriftSummary:
    status: str  # "ok", "drift_detected", "insufficient_data", "error"
    reference_count: int
    current_count: int
    number_of_columns: int
    number_of_drifted_columns: int
    share_of_drifted_columns: float
    dataset_drift: bool
    drifted_features: list[str]
    details: dict[str, Any]


def run_drift_analysis(
    bundle: LoadedBundle,
    current_df: pd.DataFrame,
    output_dir: str | Path | None = None,
    min_engines_required: int = 30,
) -> DriftSummary:
    """
    Execute drift analysis comparing current cohort features against the frozen reference cohort.
    
    Rules:
    - Reference cohort loaded from bundle.bundle_dir / "monitoring_reference.parquet".
    - If current cohort has fewer than min_engines_required (30) unique engines:
      return status="insufficient_data" without false alarms.
    - Dataset drift threshold is set to 50% (drift_share=0.50) of active feature columns.
    - Individual feature drift is detected and listed.
    - If output_dir is provided: saves drift_report.html and drift_summary.json.
    """
    ref_path = bundle.bundle_dir / "monitoring_reference.parquet"
    if not ref_path.exists():
        raise FileNotFoundError(f"Monitoring reference parquet missing in {bundle.bundle_dir}")

    ref_df = pd.read_parquet(ref_path)
    feature_cols = bundle.feature_schema["rul_feature_names"]

    n_ref = int(ref_df["unit_id"].nunique()) if "unit_id" in ref_df.columns else len(ref_df)
    n_curr = int(current_df["unit_id"].nunique()) if "unit_id" in current_df.columns else len(current_df)

    if n_curr < min_engines_required:
        summary = DriftSummary(
            status="insufficient_data",
            reference_count=n_ref,
            current_count=n_curr,
            number_of_columns=len(feature_cols),
            number_of_drifted_columns=0,
            share_of_drifted_columns=0.0,
            dataset_drift=False,
            drifted_features=[],
            details={
                "message": (
                    f"Current sample size ({n_curr} engines) is below statistical reliability threshold "
                    f"({min_engines_required} engines). Drift analysis skipped to prevent false alarms."
                ),
            },
        )
        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            (out_p / "drift_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        return summary

    # Filter to active model features
    ref_sub = ref_df[feature_cols].copy()
    curr_sub = current_df[feature_cols].copy()

    # Run Evidently DataDriftPreset with 50% dataset drift threshold
    report = Report(metrics=[DataDriftPreset(drift_share=0.50)])
    report.run(reference_data=ref_sub, current_data=curr_sub)
    rep_dict = report.as_dict()

    drift_res = {}
    drift_by_col = {}
    for m in rep_dict.get("metrics", []):
        if "drift_by_columns" in m.get("result", {}):
            drift_res = m["result"]
            drift_by_col = drift_res["drift_by_columns"]
            break
        elif "dataset_drift" in m.get("result", {}):
            drift_res = m["result"]

    n_cols = int(drift_res.get("number_of_columns", len(feature_cols)))
    n_drifted = int(drift_res.get("number_of_drifted_columns", 0))
    share_drifted = float(drift_res.get("share_of_drifted_columns", float(n_drifted) / float(n_cols) if n_cols > 0 else 0.0))
    is_dataset_drift = bool(drift_res.get("dataset_drift", share_drifted >= 0.50))

    drifted_features = []
    feature_drift_details = {}
    for col_name, c_info in drift_by_col.items():
        drifted = bool(c_info.get("drift_detected", False))
        score = float(c_info.get("drift_score", 0.0))
        stat_test = str(c_info.get("stat_test", "unknown"))
        feature_drift_details[col_name] = {
            "drift_detected": drifted,
            "drift_score": score,
            "stat_test": stat_test,
        }
        if drifted:
            drifted_features.append(col_name)

    status = "drift_detected" if (is_dataset_drift or n_drifted > 0) else "ok"

    summary = DriftSummary(
        status=status,
        reference_count=n_ref,
        current_count=n_curr,
        number_of_columns=n_cols,
        number_of_drifted_columns=n_drifted,
        share_of_drifted_columns=share_drifted,
        dataset_drift=is_dataset_drift,
        drifted_features=drifted_features,
        details={
            "feature_drift": feature_drift_details,
        },
    )

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "drift_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        try:
            report.save_html(str(out_p / "drift_report.html"))
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(f"Could not write Evidently HTML report: {e}")

    return summary


def run_drift_controls(
    bundle: LoadedBundle,
    output_dir: str | Path = "reports/drift/controls",
) -> dict[str, Any]:
    """
    Execute three controlled drift verification scenarios:
    1. No-change control: reference vs identical reference. (Expected: 0 drift, dataset_drift=False)
    2. Shifted-sensor control: reference with synthetic sensor perturbations (+2 sigma). (Expected: status='drift_detected', perturbed sensors detected)
    3. Insufficient-data control: sub-sample of 10 engines. (Expected: status='insufficient_data')
    """
    ref_path = bundle.bundle_dir / "monitoring_reference.parquet"
    ref_df = pd.read_parquet(ref_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. No-change Control
    no_change_summary = run_drift_analysis(bundle, ref_df, min_engines_required=30)

    # 2. Shifted-sensor Control (+2 sigma shift on key sensor features)
    shifted_df = ref_df.copy()
    perturbed_sensors = ["s02_last", "s03_last", "s04_last", "s07_last", "s11_last"]
    for col in perturbed_sensors:
        if col in shifted_df.columns:
            std = float(shifted_df[col].std()) if shifted_df[col].std() > 0 else 1.0
            shifted_df[col] = shifted_df[col] + 2.0 * std

    shifted_summary = run_drift_analysis(bundle, shifted_df, min_engines_required=30)

    # 3. Insufficient-data Control
    small_df = ref_df.iloc[:10].copy()
    insufficient_summary = run_drift_analysis(bundle, small_df, min_engines_required=30)

    control_results = {
        "bundle_version": bundle.version,
        "controls": {
            "no_change_control": {
                "expected_status": "ok",
                "actual_status": no_change_summary.status,
                "dataset_drift": no_change_summary.dataset_drift,
                "drifted_features_count": no_change_summary.number_of_drifted_columns,
                "passed": bool(no_change_summary.status == "ok" and not no_change_summary.dataset_drift and no_change_summary.number_of_drifted_columns == 0),
            },
            "shifted_sensor_control": {
                "perturbed_columns": perturbed_sensors,
                "expected_status": "drift_detected",
                "actual_status": shifted_summary.status,
                "dataset_drift": shifted_summary.dataset_drift,
                "drifted_features_detected": shifted_summary.drifted_features,
                "passed": bool(shifted_summary.status == "drift_detected" and any(c in shifted_summary.drifted_features for c in perturbed_sensors)),
            },
            "insufficient_data_control": {
                "sample_size": len(small_df),
                "expected_status": "insufficient_data",
                "actual_status": insufficient_summary.status,
                "passed": bool(insufficient_summary.status == "insufficient_data"),
            },
        },
        "all_controls_passed": bool(
            (no_change_summary.status == "ok" and not no_change_summary.dataset_drift)
            and (shifted_summary.status == "drift_detected")
            and (insufficient_summary.status == "insufficient_data")
        ),
    }

    (out_path / "control_results.json").write_text(json.dumps(control_results, indent=2), encoding="utf-8")
    logger.info(f"Saved drift control results to {out_path / 'control_results.json'}")
    return control_results
