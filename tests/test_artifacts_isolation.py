"""Tests for artifact immutability, test isolation, and promotion verification."""

import json
from pathlib import Path

import joblib
import pytest

from turbineguard.anomaly import fit_isolation_forest_component
from turbineguard.artifacts import compute_file_sha256, freeze_model_bundle
from turbineguard.config import Config, ModelsConfig


def test_synthetic_fitting_does_not_modify_staging(tmp_path):
    """Ensure fitting on synthetic data with isolated config does not touch real models/staging."""
    real_staging = Path("models/staging")
    staging_hashes_before = {}
    if real_staging.exists():
        for f in real_staging.iterdir():
            if f.is_file():
                staging_hashes_before[f.name] = compute_file_sha256(f)

    # Fit synthetic model to tmp_path
    isolated_staging = tmp_path / "staging"
    cfg = Config(models=ModelsConfig(staging_dir=str(isolated_staging)))

    from tests.test_anomaly import create_synthetic_proxy_dev_df
    df_synth = create_synthetic_proxy_dev_df(35)
    _pipe, _fit_res = fit_isolation_forest_component(df_synth, cfg)

    assert (isolated_staging / "anomaly_pipeline.joblib").exists()
    assert (isolated_staging / "anomaly_metadata.json").exists()

    # Verify real staging files were NOT modified
    if real_staging.exists():
        for f in real_staging.iterdir():
            if f.is_file():
                assert f.name in staging_hashes_before
                assert compute_file_sha256(f) == staging_hashes_before[f.name]


def test_freeze_rejects_overwriting_existing_frozen_bundle(tmp_path):
    """Ensure freeze_model_bundle refuses to overwrite an existing frozen version."""
    # Create dummy bundle in tmp_path
    bundle_ver = "v_test_freeze"
    bundle_dir = Path("models") / bundle_ver
    bundle_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = bundle_dir / "test.txt"
    dummy_file.write_text("existing content", encoding="utf-8")

    try:
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            freeze_model_bundle(version=bundle_ver, allow_overwrite=False)
    finally:
        # Cleanup
        if dummy_file.exists():
            dummy_file.unlink()
        if bundle_dir.exists():
            bundle_dir.rmdir()


def test_promotion_status_requires_both_g3_and_g4_pass(tmp_path):
    """Verify that promotion_status is 'promoted' ONLY when both G3 and G4 are PASS."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy pipeline and metadata
    pipe_path = staging_dir / "rul_pipeline.joblib"
    ano_path = staging_dir / "anomaly_pipeline.joblib"
    joblib.dump({"dummy": "rul"}, pipe_path)
    joblib.dump({"dummy": "ano"}, ano_path)

    champ_info = {
        "champion_experiment_id": "E05",
        "model_family": "xgboost",
        "description": "test",
        "best_params": {},
        "feature_names": ["cycle"],
        "mean_cv_rmse": 14.5,
        "std_err_cv_rmse": 0.5,
    }
    (staging_dir / "champion_info.json").write_text(json.dumps(champ_info), encoding="utf-8")

    ano_meta = {
        "n_contributing_engines": 80,
        "n_sampled_rows": 3000,
        "feature_names": ["s02_last"],
        "frozen_cutoff_99th": 0.5,
        "min_score": 0.1,
        "max_score": 0.9,
        "mean_score": 0.3,
        "pipeline_path": str(ano_path),
    }
    (staging_dir / "anomaly_metadata.json").write_text(json.dumps(ano_meta), encoding="utf-8")

    # Mock validation report with G4 FAIL
    val_report = {
        "dataset_id": "FD001",
        "evaluated_model_hashes": {
            "rul_pipeline_sha256": compute_file_sha256(pipe_path),
            "anomaly_pipeline_sha256": compute_file_sha256(ano_path),
        },
        "gates": {
            "G3_RUL_promotion": {"outcome": "PASS"},
            "G4_Anomaly_promotion": {"outcome": "FAIL"},
        },
    }
    val_path = Path("reports/validation_metrics.json")
    val_backup = val_path.read_text(encoding="utf-8") if val_path.exists() else None

    try:
        val_path.parent.mkdir(parents=True, exist_ok=True)
        val_path.write_text(json.dumps(val_report), encoding="utf-8")

        test_bundle_ver = "v_test_g4_fail"
        cfg_test = Config(models=ModelsConfig(staging_dir=str(staging_dir)))
        cfg_file = tmp_path / "cfg.yaml"
        import yaml
        cfg_file.write_text(yaml.safe_dump(cfg_test.model_dump()), encoding="utf-8")

        frozen_p = freeze_model_bundle(config_path=cfg_file, version=test_bundle_ver, allow_overwrite=True)
        meta = json.loads((frozen_p / "metadata.json").read_text(encoding="utf-8"))

        # Since G4 failed, promotion status MUST be experimental
        assert meta["promotion_status"] == "experimental"

    finally:
        # Cleanup
        if val_backup is not None:
            val_path.write_text(val_backup, encoding="utf-8")
        test_bundle_dir = Path("models") / "v_test_g4_fail"
        if test_bundle_dir.exists():
            import shutil
            shutil.rmtree(test_bundle_dir)
