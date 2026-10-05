"""Unit tests for labels, engine splits, and frozen snapshot manifests."""

import numpy as np
import pandas as pd

from turbineguard.labels import assign_anomaly_proxies, compute_rul_labels
from turbineguard.splits import create_engine_splits, create_snapshot_manifest


def create_synthetic_train_df(n_engines: int = 10, min_len: int = 35, max_len: int = 150) -> pd.DataFrame:
    """Create synthetic training trajectories of variable lengths."""
    rng = np.random.default_rng(42)
    records = []
    for u in range(1, n_engines + 1):
        length = int(rng.integers(min_len, max_len + 1))
        for c in range(1, length + 1):
            row = {"unit_id": u, "cycle": c, "op_1": 0.0, "op_2": 0.0, "op_3": 100.0}
            for s in range(1, 22):
                row[f"s{s:02d}"] = float(s * 5 + c * 0.05)
            records.append(row)
    df = pd.DataFrame(records)
    df["source_partition"] = "train"
    return df


def test_compute_rul_labels():
    """Test RUL label computation on known lifetimes."""
    df = pd.DataFrame({
        "unit_id": [1, 1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 4, 1, 2],
    })
    labeled = compute_rul_labels(df, target_cap=2)
    
    # Engine 1: T=4 -> RULs: 3, 2, 1, 0. Capped (cap=2): 2, 2, 1, 0.
    e1_true = labeled[labeled["unit_id"] == 1]["rul_true"].tolist()
    e1_capped = labeled[labeled["unit_id"] == 1]["rul_capped"].tolist()
    assert e1_true == [3, 2, 1, 0]
    assert e1_capped == [2.0, 2.0, 1.0, 0.0]
    
    # Terminal RUL is 0
    assert e1_true[-1] == 0
    
    # Engine 2: T=2 -> RULs: 1, 0.
    e2_true = labeled[labeled["unit_id"] == 2]["rul_true"].tolist()
    assert e2_true == [1, 0]


def test_assign_anomaly_proxies():
    """Test anomaly proxy categories (high RUL >= 100, near failure <= 30, intermediate)."""
    df = pd.DataFrame({
        "rul_true": [150, 100, 99, 31, 30, 10, 0],
    })
    df_proxies = assign_anomaly_proxies(df, high_threshold=100, near_failure_threshold=30)
    
    assert df_proxies["proxy_high_rul"].tolist() == [True, True, False, False, False, False, False]
    assert df_proxies["proxy_near_failure"].tolist() == [False, False, False, False, True, True, True]
    assert df_proxies["proxy_intermediate"].tolist() == [False, False, True, True, False, False, False]


def test_create_engine_splits_disjointness_and_folds():
    """Test 80/20 engine split and 5-fold grouped CV."""
    engine_ids = list(range(1, 101))
    split = create_engine_splits(engine_ids, dev_count=80, val_count=20, n_folds=5, seed=42)
    
    assert len(split.dev_engines) == 80
    assert len(split.val_engines) == 20
    assert len(set(split.dev_engines).intersection(set(split.val_engines))) == 0
    
    # Check 5 folds
    assert len(split.cv_folds) == 5
    all_fold_engines = []
    for fold_engines in split.cv_folds.values():
        assert len(fold_engines) == 16  # 80 / 5 = 16
        all_fold_engines.extend(fold_engines)
    
    # Union of all folds must equal dev engines
    assert sorted(all_fold_engines) == sorted(split.dev_engines)
    # Check deterministic hash
    split2 = create_engine_splits(engine_ids, dev_count=80, val_count=20, n_folds=5, seed=42)
    assert split.split_hash == split2.split_hash


def test_create_snapshot_manifest():
    """Test snapshot cutoffs for validation engines."""
    df = create_synthetic_train_df(n_engines=20, min_len=120, max_len=200)
    val_engines = list(range(1, 21))
    
    manifest = create_snapshot_manifest(df, val_engines, offsets=[10, 30, 60, 90], min_history=20, seed=42)
    
    assert len(manifest.val_snapshots) == 20
    for snap in manifest.val_snapshots.values():
        assert snap["lifetime_T"] - snap["assigned_offset"] == snap["cut_cycle"]
        assert snap["cut_cycle"] >= 20
        assert snap["true_rul_at_cut"] == snap["assigned_offset"]
