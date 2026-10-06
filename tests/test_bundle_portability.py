"""A packaging revision preserves predictions and exact hashes across platforms."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbineguard.artifacts import (
    compute_file_sha256,
    load_model_bundle,
    repackage_model_bundle,
)


def test_packaging_revision_preserves_models_and_canonical_hashes(tmp_path):
    source = Path("models/v0.2.1")
    original = load_model_bundle(source)
    source_hashes = {p.name: compute_file_sha256(p) for p in source.iterdir() if p.is_file()}
    target = tmp_path / "portable"
    repackage_model_bundle(source, target, version="test-packaging")
    revised = load_model_bundle(target)

    assert revised.version == "test-packaging"
    assert revised.metadata["packaging_revision"]["source_bundle_version"] == "v0.2.1"
    for filename in revised.metadata["file_hashes"]:
        assert compute_file_sha256(target / filename) == revised.metadata["file_hashes"][filename]
        if filename.endswith(".json"):
            assert b"\r" not in (target / filename).read_bytes()
            assert json.loads((target / filename).read_bytes()) == json.loads((source / filename).read_bytes())
    reference = pd.read_parquet(target / "monitoring_reference.parquet")
    features = reference[original.feature_schema["rul_feature_names"]]
    np.testing.assert_allclose(original.rul_pipeline.predict(features), revised.rul_pipeline.predict(features), atol=1e-6)
    for name, digest in source_hashes.items():
        assert compute_file_sha256(source / name) == digest

    with pytest.raises(FileExistsError):
        repackage_model_bundle(source, target, version="v0.2.1")
    policy = json.loads((target / "policy.json").read_bytes())
    policy["horizon_cycles"] += 1
    (target / "policy.json").write_bytes(json.dumps(policy).encode("utf-8"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model_bundle(target)


def test_checked_in_release_has_exact_byte_hashes():
    """The public release must load directly without checkout newline repair."""
    bundle = load_model_bundle("models/v0.2.1")
    assert bundle.version == "v0.2.1"
