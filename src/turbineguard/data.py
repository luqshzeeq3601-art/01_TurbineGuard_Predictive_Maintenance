"""Data loading, schema validation, and quality assessment for NASA C-MAPSS FD001."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "unit_id",
    "cycle",
    "op_1",
    "op_2",
    "op_3",
    "s01",
    "s02",
    "s03",
    "s04",
    "s05",
    "s06",
    "s07",
    "s08",
    "s09",
    "s10",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s16",
    "s17",
    "s18",
    "s19",
    "s20",
    "s21",
]

SENSOR_COLUMNS = [f"s{i:02d}" for i in range(1, 22)]
SETTING_COLUMNS = ["op_1", "op_2", "op_3"]


class ValidationError(ValueError):
    """Raised when data fails schema, chronology, or integrity validation."""


@dataclass
class QualityReport:
    """Structured data quality report."""
    dataset_id: str
    train_rows: int
    train_engines: int
    test_rows: int
    test_engines: int
    test_rul_count: int
    columns_count: int
    has_nulls: bool
    has_infinite: bool
    duplicate_keys_count: int
    chronology_valid: bool
    constant_sensors_train: list[str]
    nonconstant_sensors_train: list[str]
    engine_cycle_stats: dict[str, dict]


def load_raw_cmapss_file(file_path: Path, partition: str = "train") -> pd.DataFrame:
    """
    Load raw whitespace-delimited C-MAPSS text file.
    
    Validates:
    - 26 columns per row.
    - No empty trailing columns created by delimiters.
    - Explicit types for unit_id and cycle.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    # Read lines or use read_csv with regex whitespace
    # Using read_csv with sep=r"\s+"
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    
    if df.shape[1] != len(RAW_COLUMNS):
        raise ValidationError(
            f"Expected exactly {len(RAW_COLUMNS)} columns, found {df.shape[1]} in {file_path}"
        )
    
    df.columns = RAW_COLUMNS
    df["source_partition"] = partition
    
    # Cast types safely
    try:
        df["unit_id"] = df["unit_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)
        for col in SETTING_COLUMNS + SENSOR_COLUMNS:
            df[col] = df[col].astype(float)
    except Exception as e:
        raise ValidationError(f"Failed to cast column types in {file_path}: {e}") from e

    return df


def load_raw_rul_file(file_path: Path) -> np.ndarray:
    """Load endpoint RUL text file containing one RUL integer/float per line."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"RUL file not found: {file_path}")

    rul_values = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split()
            if len(parts) != 1:
                raise ValidationError(
                    f"Expected single RUL value per line, got {len(parts)} at line {line_num} in {file_path}"
                )
            try:
                val = float(parts[0])
                if np.isnan(val) or np.isinf(val) or val < 0:
                    raise ValidationError(f"Invalid RUL value {val} at line {line_num} in {file_path}")
                rul_values.append(val)
            except ValueError as e:
                raise ValidationError(f"Non-numeric RUL value at line {line_num} in {file_path}: {e}") from e

    return np.array(rul_values, dtype=float)


def validate_trajectory_chronology(df: pd.DataFrame, partition: str = "train") -> None:
    """
    Validate trajectory chronology:
    - Per-engine cycles start at 1.
    - Cycles increase strictly by 1 with no gaps and no duplicates.
    - No duplicate (source_partition, unit_id, cycle) keys.
    """
    # Check duplicate keys
    dups = df.duplicated(subset=["source_partition", "unit_id", "cycle"])
    if dups.any():
        num_dups = dups.sum()
        dup_rows = df[dups].head(3).to_dict(orient="records")
        raise ValidationError(
            f"Found {num_dups} duplicate (source_partition, unit_id, cycle) keys in {partition} data: {dup_rows}"
        )

    # Check per-engine chronology
    for unit_id, group in df.groupby("unit_id", sort=False):
        cycles = group["cycle"].values
        if len(cycles) == 0:
            raise ValidationError(f"Engine {unit_id} in {partition} has empty trajectory.")
        
        # Check start cycle
        if cycles[0] != 1:
            raise ValidationError(
                f"Engine {unit_id} in {partition} starts at cycle {cycles[0]}, expected cycle 1."
            )
        
        # Check step = 1 and strictly increasing
        diffs = np.diff(cycles)
        if not np.all(diffs == 1):
            bad_indices = np.where(diffs != 1)[0]
            step_err = diffs[bad_indices[0]]
            from_cycle = cycles[bad_indices[0]]
            to_cycle = cycles[bad_indices[0] + 1]
            raise ValidationError(
                f"Engine {unit_id} in {partition} has invalid cycle step {step_err} "
                f"(from cycle {from_cycle} to {to_cycle}). Cycles must be strictly consecutive."
            )


def validate_finite_values(df: pd.DataFrame, partition: str = "train") -> None:
    """Validate that all numeric columns contain finite, non-null values."""
    null_counts = df.isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        raise ValidationError(f"Null values detected in {partition} data: {cols_with_nulls}")

    for col in SETTING_COLUMNS + SENSOR_COLUMNS:
        values = df[col].values
        if not np.all(np.isfinite(values)):
            raise ValidationError(f"Non-finite values (inf or -inf) detected in column {col} of {partition} data.")


def detect_constant_columns(df: pd.DataFrame, columns: list[str], tol: float = 1e-12) -> tuple[list[str], list[str]]:
    """Identify constant columns (zero standard deviation) on a dataset."""
    constant_cols = []
    nonconstant_cols = []
    for col in columns:
        std_val = df[col].std(ddof=0)
        if np.isnan(std_val) or std_val <= tol:
            constant_cols.append(col)
        else:
            nonconstant_cols.append(col)
    return constant_cols, nonconstant_cols


def run_data_quality_checks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_array: np.ndarray,
) -> QualityReport:
    """
    Run comprehensive quality checks on raw FD001 data and return QualityReport.
    """
    # 1. Finite checks
    validate_finite_values(train_df, "train")
    validate_finite_values(test_df, "test")

    # 2. Chronology checks
    validate_trajectory_chronology(train_df, "train")
    validate_trajectory_chronology(test_df, "test")

    # 3. Test RUL count check
    test_engines = test_df["unit_id"].nunique()
    if len(rul_array) != test_engines:
        raise ValidationError(
            f"Mismatch between test engine count ({test_engines}) and RUL count ({len(rul_array)})."
        )

    # 4. Detect constant columns on train
    constant_sensors, nonconstant_sensors = detect_constant_columns(train_df, SENSOR_COLUMNS)

    # 5. Engine stats
    train_lengths = train_df.groupby("unit_id")["cycle"].max()
    test_lengths = test_df.groupby("unit_id")["cycle"].max()

    report = QualityReport(
        dataset_id="FD001",
        train_rows=len(train_df),
        train_engines=int(train_df["unit_id"].nunique()),
        test_rows=len(test_df),
        test_engines=test_engines,
        test_rul_count=len(rul_array),
        columns_count=len(RAW_COLUMNS),
        has_nulls=False,
        has_infinite=False,
        duplicate_keys_count=0,
        chronology_valid=True,
        constant_sensors_train=constant_sensors,
        nonconstant_sensors_train=nonconstant_sensors,
        engine_cycle_stats={
            "train_min_cycles": int(train_lengths.min()),
            "train_max_cycles": int(train_lengths.max()),
            "train_mean_cycles": float(train_lengths.mean()),
            "test_min_cycles": int(test_lengths.min()),
            "test_max_cycles": int(test_lengths.max()),
            "test_mean_cycles": float(test_lengths.mean()),
        },
    )
    return report


def save_quality_report(report: QualityReport, output_path: Path) -> None:
    """Save QualityReport as formatted JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
