"""Public-checkout integration fixtures without local NASA data or old bundles."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from turbineguard.artifacts import load_model_bundle
from turbineguard.data import RAW_COLUMNS


def _history(units: int, cycles: int) -> pd.DataFrame:
    rows = []
    for unit in range(1, units + 1):
        for cycle in range(1, cycles + 1):
            row = {"unit_id": unit, "cycle": cycle, "op_1": -0.0005,
                   "op_2": 0.0002, "op_3": 100.0}
            row.update({f"s{sensor:02d}": float(sensor * 10 + cycle * 0.1 + unit)
                        for sensor in range(1, 22)})
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def portable_bundle_dir(tmp_path_factory):
    source = Path(__file__).resolve().parents[1] / "models" / "v0.2.1"
    destination = tmp_path_factory.mktemp("trusted-bundle") / "bundle"
    shutil.copytree(source, destination)
    load_model_bundle(destination)
    return destination


@pytest.fixture(scope="session")
def loaded_bundle(portable_bundle_dir):
    return load_model_bundle(portable_bundle_dir)


@pytest.fixture
def synthetic_history():
    return _history(5, 31)


@pytest.fixture
def isolated_freeze_inputs(tmp_path, monkeypatch):
    """Exercise freezing under a temporary working directory with synthetic inputs."""
    monkeypatch.chdir(tmp_path)
    raw_dir = tmp_path / "data" / "raw" / "FD001"
    raw_dir.mkdir(parents=True)
    _history(80, 120)[RAW_COLUMNS].to_csv(raw_dir / "train_FD001.txt", sep=" ", header=False, index=False)
    processed = tmp_path / "data" / "processed"
    processed.mkdir()
    (processed / "split_manifest.json").write_text(json.dumps({"dev_engines": list(range(1, 81))}))
    (tmp_path / "reports").mkdir()
    return tmp_path
