"""Engine splitting, group cross-validation, and frozen snapshot manifests."""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


@dataclass
class SplitManifest:
    """Manifest of engine splits and cross-validation fold assignments."""
    dataset_id: str
    seed: int
    rng_version: str
    dev_engines: list[int]
    val_engines: list[int]
    cv_folds: dict[str, list[int]]  # fold_0 to fold_4 engine IDs
    split_hash: str


@dataclass
class SnapshotManifest:
    """Manifest of frozen validation and evaluation snapshot cutoffs."""
    dataset_id: str
    seed: int
    val_snapshots: dict[str, dict]  # unit_id -> {lifetime_T, offset, cut_cycle, true_rul_at_cut}
    dev_eval_offsets: list[int]
    snapshot_hash: str


def compute_dict_hash(d: dict) -> str:
    """Compute deterministic SHA-256 hash of a dictionary."""
    encoded = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_engine_splits(
    engine_ids: list[int],
    dev_count: int = 80,
    val_count: int = 20,
    n_folds: int = 5,
    seed: int = 42,
) -> SplitManifest:
    """
    Split engine IDs into development and validation sets and create grouped CV folds.
    
    1. Sort engine IDs and permute with np.random.default_rng(seed).
    2. First dev_count are dev, remaining val_count are validation.
    3. Split dev engines into n_folds using GroupKFold.
    """
    sorted_ids = sorted(set(engine_ids))
    if len(sorted_ids) != dev_count + val_count:
        raise ValueError(
            f"Total engines ({len(sorted_ids)}) != dev_count ({dev_count}) + val_count ({val_count})"
        )

    rng = np.random.default_rng(seed)
    permuted = rng.permutation(sorted_ids).tolist()

    dev_engines = sorted(permuted[:dev_count])
    val_engines = sorted(permuted[dev_count:])

    # Verify disjointness
    assert len(set(dev_engines).intersection(set(val_engines))) == 0

    # GroupKFold on dev engines
    gkf = GroupKFold(n_splits=n_folds)
    # Dummy X, y to split groups
    dev_arr = np.array(dev_engines)
    cv_folds = {}
    for fold_idx, (_, test_indices) in enumerate(gkf.split(dev_arr, groups=dev_arr)):
        cv_folds[f"fold_{fold_idx}"] = sorted(dev_arr[test_indices].tolist())

    manifest_dict = {
        "dataset_id": "FD001",
        "seed": seed,
        "rng_version": f"numpy_{np.__version__}",
        "dev_engines": dev_engines,
        "val_engines": val_engines,
        "cv_folds": cv_folds,
    }
    split_hash = compute_dict_hash(manifest_dict)

    return SplitManifest(
        dataset_id="FD001",
        seed=seed,
        rng_version=f"numpy_{np.__version__}",
        dev_engines=dev_engines,
        val_engines=val_engines,
        cv_folds=cv_folds,
        split_hash=split_hash,
    )


def create_snapshot_manifest(
    train_df: pd.DataFrame,
    val_engines: list[int],
    offsets: list[int] | None = None,
    min_history: int = 20,
    seed: int = 42,
) -> SnapshotManifest:
    """
    Create frozen validation snapshot cutoffs.
    
    1. Permute val_engines with seed.
    2. Cyclically assign offsets [10, 30, 60, 90].
    3. cut_cycle = T - offset. Require cut_cycle >= min_history.
    4. If impossible, choose largest eligible offset from list and log exception.
    """
    offsets = offsets or [10, 30, 60, 90]
    rng = np.random.default_rng(seed)
    permuted_val = rng.permutation(val_engines).tolist()

    lifetimes = train_df.groupby("unit_id")["cycle"].max().to_dict()

    val_snapshots = {}
    for idx, unit_id in enumerate(permuted_val):
        t_max = lifetimes[unit_id]
        chosen_offset = offsets[idx % len(offsets)]
        cut_cycle = t_max - chosen_offset

        if cut_cycle < min_history:
            # Fallback to largest eligible offset
            eligible = [off for off in offsets if t_max - off >= min_history]
            if not eligible:
                raise ValueError(
                    f"Validation engine {unit_id} (lifetime {t_max}) cannot satisfy min_history {min_history} with any offset."
                )
            chosen_offset = max(eligible)
            cut_cycle = t_max - chosen_offset
            logger.warning(
                f"Validation engine {unit_id} (T={t_max}): requested offset was adjusted to {chosen_offset} (cut_cycle={cut_cycle})"
            )

        val_snapshots[str(unit_id)] = {
            "unit_id": int(unit_id),
            "lifetime_T": int(t_max),
            "assigned_offset": int(chosen_offset),
            "cut_cycle": int(cut_cycle),
            "true_rul_at_cut": int(chosen_offset),
        }

    manifest_dict = {
        "dataset_id": "FD001",
        "seed": seed,
        "val_snapshots": val_snapshots,
        "dev_eval_offsets": offsets,
    }
    snapshot_hash = compute_dict_hash(manifest_dict)

    return SnapshotManifest(
        dataset_id="FD001",
        seed=seed,
        val_snapshots=val_snapshots,
        dev_eval_offsets=offsets,
        snapshot_hash=snapshot_hash,
    )


def save_split_manifest(manifest: SplitManifest, output_path: Path) -> None:
    """Save split manifest to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)


def save_snapshot_manifest(manifest: SnapshotManifest, output_path: Path) -> None:
    """Save snapshot manifest to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)
