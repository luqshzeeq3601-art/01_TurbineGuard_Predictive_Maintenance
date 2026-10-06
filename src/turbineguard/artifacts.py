"""Model bundle packaging, artifact integrity verification, and bundle loader."""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from turbineguard.config import load_config
from turbineguard.data import (
    RAW_COLUMNS,
    load_raw_cmapss_file,
)
from turbineguard.features import WINDOW_SIZE, extract_snapshot_features
from turbineguard.labels import compute_rul_labels

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_bundle_json(path: Path, value: dict) -> None:
    """Write deterministic UTF-8/LF bytes before recording integrity hashes."""
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


@dataclass
class LoadedBundle:
    """Container for loaded and verified model bundle components."""
    version: str
    bundle_dir: Path
    rul_pipeline: Any
    anomaly_pipeline: Any
    policy: dict
    feature_schema: dict
    metadata: dict


def freeze_model_bundle(
    config_path: str | Path = "configs/default.yaml",
    version: str = "v0.1.0",
    allow_overwrite: bool = False,
) -> Path:
    """
    Freeze validated staging models and configuration into an immutable versioned bundle directory:
    - models/<version>/rul_pipeline.joblib
    - models/<version>/anomaly_pipeline.joblib
    - models/<version>/policy.json
    - models/<version>/feature_schema.json
    - models/<version>/monitoring_reference.parquet & reference_manifest.json
    - models/<version>/metadata.json
    """
    cfg = load_config(config_path)
    staging_dir = Path(cfg.models.staging_dir)
    bundle_dir = Path("models") / version

    if bundle_dir.exists() and any(bundle_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(
            f"Frozen bundle directory {bundle_dir} already exists and is non-empty. "
            f"Refusing to overwrite immutable frozen release {version}."
        )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    staging_rul = staging_dir / "rul_pipeline.joblib"
    staging_champ_info = staging_dir / "champion_info.json"
    staging_ano = staging_dir / "anomaly_pipeline.joblib"
    staging_ano_meta = staging_dir / "anomaly_metadata.json"

    if not (staging_rul.exists() and staging_ano.exists() and staging_champ_info.exists() and staging_ano_meta.exists()):
        raise FileNotFoundError(f"Missing staged models or metadata in {staging_dir}. Run training first.")

    champ_info = json.loads(staging_champ_info.read_text(encoding="utf-8"))
    ano_meta = json.loads(staging_ano_meta.read_text(encoding="utf-8"))

    staging_rul_hash = compute_file_sha256(staging_rul)
    staging_ano_hash = compute_file_sha256(staging_ano)

    # Load validation report for gate outcomes and provenance verification
    val_metrics_file = Path("reports/validation_metrics.json")
    if not val_metrics_file.exists():
        raise FileNotFoundError(
            f"Validation report missing at {val_metrics_file}. "
            "Validation evaluation must be executed on staged models prior to freezing."
        )
    val_info = json.loads(val_metrics_file.read_text(encoding="utf-8"))

    # Verify that validation metrics evaluated the EXACT staged models
    evaluated_hashes = val_info.get("evaluated_model_hashes", {})
    if evaluated_hashes:
        eval_rul_hash = evaluated_hashes.get("rul_pipeline_sha256")
        eval_ano_hash = evaluated_hashes.get("anomaly_pipeline_sha256")
        if eval_rul_hash != staging_rul_hash:
            raise ValueError(
                f"Stale validation report: evaluated RUL model hash ({eval_rul_hash}) "
                f"does not match staged RUL model hash ({staging_rul_hash})."
            )
        if eval_ano_hash != staging_ano_hash:
            raise ValueError(
                f"Stale validation report: evaluated anomaly model hash ({eval_ano_hash}) "
                f"does not match staged anomaly model hash ({staging_ano_hash})."
            )

    # 1. Copy pipelines
    dest_rul = bundle_dir / "rul_pipeline.joblib"
    dest_ano = bundle_dir / "anomaly_pipeline.joblib"
    dest_rul.write_bytes(staging_rul.read_bytes())
    dest_ano.write_bytes(staging_ano.read_bytes())

    # 2. Write policy.json
    policy_dict = {
        "policy_version": "1.0",
        "horizon_cycles": cfg.policy.horizon_cycles,
        "capacity_fraction": cfg.policy.capacity_fraction,
        "anomaly_cutoff": ano_meta["frozen_cutoff_99th"],
        "anomaly_score_direction": "higher_is_more_anomalous",
        "priority_rules": {
            "review_soon": "within_horizon == True",
            "investigate": "within_horizon == False and anomaly_flag == True",
            "routine_review": "otherwise",
        },
        "sort_order": ["estimated_rul ASC", "anomaly_flag DESC", "unit_id ASC"],
    }
    _write_bundle_json(bundle_dir / "policy.json", policy_dict)

    # 3. Write feature_schema.json
    schema_dict = {
        "schema_version": "1.0",
        "window_size": WINDOW_SIZE,
        "min_history": cfg.features.min_history,
        "target_cap": cfg.features.target_cap,
        "raw_columns": RAW_COLUMNS,
        "rul_feature_names": champ_info["feature_names"],
        "anomaly_feature_names": ano_meta["feature_names"],
        "constant_sensors_dropped": ["s01", "s05", "s10", "s16", "s18", "s19"],
    }
    _write_bundle_json(bundle_dir / "feature_schema.json", schema_dict)

    # 4. Generate frozen monitoring reference features (80 dev engine snapshots)
    split_manifest = json.loads(Path(cfg.data.split_manifest).read_text(encoding="utf-8"))
    dev_engines = split_manifest["dev_engines"]
    raw_train = load_raw_cmapss_file(Path(cfg.data.raw_dir) / "train_FD001.txt", "train")
    dev_raw = raw_train[raw_train["unit_id"].isin(dev_engines)].copy()
    dev_labeled = compute_rul_labels(dev_raw, target_cap=cfg.features.target_cap)

    # Assign each dev engine one snapshot using cyclical offsets [10, 30, 60, 90]
    dev_lifetimes = dev_labeled.groupby("unit_id")["cycle"].max().to_dict()
    offsets = cfg.splits.val_snapshot_offsets
    dev_snapshots = {}
    for idx, uid in enumerate(dev_engines):
        t_max = dev_lifetimes[uid]
        off = offsets[idx % len(offsets)]
        cut_c = t_max - off
        if cut_c < 20:
            off = max([o for o in offsets if t_max - o >= 20])
            cut_c = t_max - off
        dev_snapshots[str(uid)] = {
            "unit_id": uid,
            "lifetime_T": t_max,
            "assigned_offset": off,
            "cut_cycle": cut_c,
            "true_rul_at_cut": off,
        }

    ref_feats = extract_snapshot_features(
        dev_labeled,
        snapshot_cutoffs=dev_snapshots,
        window_size=cfg.features.window_size,
        target_cap=cfg.features.target_cap,
    )
    ref_parquet_path = bundle_dir / "monitoring_reference.parquet"
    ref_feats.to_parquet(ref_parquet_path, index=False)

    ref_manifest = {
        "n_engines": len(ref_feats),
        "sampling_policy": "one_snapshot_per_development_engine",
        "offsets_used": offsets,
        "feature_count": len(ref_feats.columns),
        "parquet_sha256": compute_file_sha256(ref_parquet_path),
    }
    _write_bundle_json(bundle_dir / "reference_manifest.json", ref_manifest)

    # 5. Compute bundle file checksums and write metadata.json
    file_hashes = {}
    for f in ["rul_pipeline.joblib", "anomaly_pipeline.joblib", "policy.json", "feature_schema.json", "monitoring_reference.parquet", "reference_manifest.json"]:
        p = bundle_dir / f
        if p.exists():
            file_hashes[f] = compute_file_sha256(p)

    # Promotion requirement: BOTH G3 (RUL) and G4 (Anomaly) MUST be PASS
    g3_pass = val_info.get("gates", {}).get("G3_RUL_promotion", {}).get("outcome") == "PASS"
    g4_pass = val_info.get("gates", {}).get("G4_Anomaly_promotion", {}).get("outcome") == "PASS"
    promotion_status = "promoted" if (g3_pass and g4_pass) else "experimental"

    metadata_dict = {
        "bundle_version": version,
        "dataset_id": "FD001",
        "frozen_timestamp_utc": datetime.now(UTC).isoformat(),
        "champion_experiment_id": champ_info["champion_experiment_id"],
        "model_family": champ_info["model_family"],
        "promotion_status": promotion_status,
        "validation_gates": val_info.get("gates", {}),
        "file_hashes": file_hashes,
        "staged_model_hashes": {
            "rul_pipeline_sha256": staging_rul_hash,
            "anomaly_pipeline_sha256": staging_ano_hash,
        },
        "source_manifest_hash": compute_file_sha256(Path(cfg.data.source_manifest)) if Path(cfg.data.source_manifest).exists() else None,
        "split_manifest_hash": compute_file_sha256(Path(cfg.data.split_manifest)) if Path(cfg.data.split_manifest).exists() else None,
        "config_seed": cfg.project.seed,
    }
    _write_bundle_json(bundle_dir / "metadata.json", metadata_dict)

    logger.info(f"Successfully frozen model bundle {version} (status={promotion_status}) to {bundle_dir}")
    return bundle_dir


def repackage_model_bundle(
    source_dir: str | Path, destination_dir: str | Path, *, version: str
) -> Path:
    """Create a portable packaging revision without training or changing model weights.

    The source must be a verified, trusted local bundle. Existing destinations are
    never overwritten and the historical source is left byte-for-byte intact.
    """
    source = load_model_bundle(source_dir)
    destination = Path(destination_dir)
    if destination.exists():
        raise FileExistsError(f"Packaging destination already exists: {destination}")
    files = {
        "rul_pipeline.joblib", "anomaly_pipeline.joblib", "policy.json",
        "feature_schema.json", "monitoring_reference.parquet", "reference_manifest.json",
    }
    if set(source.metadata["file_hashes"]) != files:
        raise ValueError("Source bundle must contain the complete expected file manifest")
    destination.mkdir(parents=True)
    for filename in sorted(files):
        source_file = source.bundle_dir / filename
        target_file = destination / filename
        if filename.endswith(".json"):
            _write_bundle_json(target_file, json.loads(source_file.read_bytes()))
        else:
            target_file.write_bytes(source_file.read_bytes())
    metadata = dict(source.metadata)
    metadata.update(
        bundle_version=version,
        file_hashes={name: compute_file_sha256(destination / name) for name in sorted(files)},
        packaging_revision={
            "source_bundle_version": source.version,
            "source_metadata_sha256": compute_file_sha256(source.bundle_dir / "metadata.json"),
            "model_weights_changed": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "json_encoding": "UTF-8 with LF newlines",
        },
    )
    _write_bundle_json(destination / "metadata.json", metadata)
    load_model_bundle(destination)
    return destination


def load_model_bundle(bundle_dir: str | Path) -> LoadedBundle:
    """
    Load and verify trusted project bundle from bundle directory.
    Validates:
    - Expected files exist.
    - Checksums match metadata.json.
    - Deserializes only trusted bundle artifacts.
    """
    bundle_path = Path(bundle_dir)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_path}")

    meta_file = bundle_path / "metadata.json"
    if not meta_file.exists():
        raise ValueError(f"Invalid bundle: missing metadata.json in {bundle_path}")

    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    file_hashes = metadata.get("file_hashes", {})

    # Verify hashes of critical files
    for fname, expected_hash in file_hashes.items():
        fpath = bundle_path / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing bundle file: {fname} in {bundle_path}")
        actual_hash = compute_file_sha256(fpath)
        if actual_hash != expected_hash:
            raise ValueError(f"Integrity error: checksum mismatch for {fname} in bundle {bundle_path}")

    # Load pipelines and configs
    rul_pipeline = joblib.load(bundle_path / "rul_pipeline.joblib")
    anomaly_pipeline = joblib.load(bundle_path / "anomaly_pipeline.joblib")
    policy = json.loads((bundle_path / "policy.json").read_text(encoding="utf-8"))
    feature_schema = json.loads((bundle_path / "feature_schema.json").read_text(encoding="utf-8"))

    return LoadedBundle(
        version=metadata.get("bundle_version", "unknown"),
        bundle_dir=bundle_path,
        rul_pipeline=rul_pipeline,
        anomaly_pipeline=anomaly_pipeline,
        policy=policy,
        feature_schema=feature_schema,
        metadata=metadata,
    )
