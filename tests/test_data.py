"""Unit tests for data acquisition, loading, schema validation, and quality assessment."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.download_data import safe_extract_zip
from turbineguard.data import (
    RAW_COLUMNS,
    SENSOR_COLUMNS,
    ValidationError,
    detect_constant_columns,
    load_raw_cmapss_file,
    load_raw_rul_file,
    run_data_quality_checks,
    validate_finite_values,
    validate_trajectory_chronology,
)


def create_synthetic_cmapss_df(
    n_units: int = 3,
    cycles_per_unit: int = 25,
    constant_sensors: list[str] | None = None,
) -> pd.DataFrame:
    """Helper to create valid synthetic C-MAPSS dataframe."""
    records = []
    constant_sensors = constant_sensors or ["s01", "s05"]
    
    for u in range(1, n_units + 1):
        for c in range(1, cycles_per_unit + 1):
            row = {
                "unit_id": u,
                "cycle": c,
                "op_1": -0.0005,
                "op_2": 0.0002,
                "op_3": 100.0,
            }
            for s_idx in range(1, 22):
                col_name = f"s{s_idx:02d}"
                if col_name in constant_sensors:
                    row[col_name] = float(s_idx * 10)
                else:
                    row[col_name] = float(s_idx * 10 + c * 0.1 + np.sin(c))
            records.append(row)
    
    df = pd.DataFrame(records)
    df["source_partition"] = "train"
    return df


def test_safe_extract_zip_prevents_path_traversal(tmp_path: Path):
    """Verify that Zip Slip path traversal attempts raise a security ValueError."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.writestr("../evil.txt", "malicious payload")
        z.writestr("good.txt", "good payload")
    
    zip_bytes = zip_buffer.getvalue()
    dest_dir = tmp_path / "safe_output"
    
    with pytest.raises(ValueError, match="Security error: Zip member"):
        safe_extract_zip(zip_bytes, dest_dir, ["evil.txt", "good.txt"])


def test_load_raw_cmapss_file_valid(tmp_path: Path):
    """Test loading a properly formatted C-MAPSS raw file."""
    synth_df = create_synthetic_cmapss_df(n_units=2, cycles_per_unit=10)
    file_path = tmp_path / "train_synth.txt"
    
    # Write whitespace separated without header
    synth_df[RAW_COLUMNS].to_csv(file_path, sep=" ", header=False, index=False)
    
    loaded_df = load_raw_cmapss_file(file_path, partition="train")
    assert loaded_df.shape == (20, len(RAW_COLUMNS) + 1)  # 26 + source_partition
    assert list(loaded_df.columns[:26]) == RAW_COLUMNS
    assert loaded_df["unit_id"].dtype == np.int64 or loaded_df["unit_id"].dtype == int


def test_load_raw_cmapss_file_bad_column_count(tmp_path: Path):
    """Test error when raw file does not have exactly 26 columns."""
    file_path = tmp_path / "bad_cols.txt"
    file_path.write_text("1 1 0.1 0.2 0.3 10.0\n", encoding="utf-8")
    
    with pytest.raises(ValidationError, match="Expected exactly 26 columns"):
        load_raw_cmapss_file(file_path, partition="train")


def test_load_raw_rul_file_valid_and_invalid(tmp_path: Path):
    """Test loading valid RUL file and rejecting invalid formatting."""
    valid_path = tmp_path / "RUL_valid.txt"
    valid_path.write_text("112\n98\n45\n", encoding="utf-8")
    
    rul = load_raw_rul_file(valid_path)
    assert len(rul) == 3
    assert np.allclose(rul, [112.0, 98.0, 45.0])
    
    # Negative RUL
    invalid_neg = tmp_path / "RUL_neg.txt"
    invalid_neg.write_text("112\n-5\n45\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Invalid RUL value"):
        load_raw_rul_file(invalid_neg)
        
    # Non-numeric RUL
    invalid_txt = tmp_path / "RUL_txt.txt"
    invalid_txt.write_text("112\nabc\n45\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Non-numeric RUL value"):
        load_raw_rul_file(invalid_txt)


def test_validate_trajectory_chronology():
    """Test cycle chronology validation (start at 1, consecutive, no gaps, no duplicates)."""
    # Valid
    df_valid = create_synthetic_cmapss_df(n_units=2, cycles_per_unit=5)
    validate_trajectory_chronology(df_valid, "train")
    
    # Start at cycle 2
    df_bad_start = df_valid.copy()
    df_bad_start.loc[df_bad_start["unit_id"] == 1, "cycle"] += 1
    with pytest.raises(ValidationError, match="starts at cycle 2, expected cycle 1"):
        validate_trajectory_chronology(df_bad_start, "train")
        
    # Cycle gap (e.g. cycle 3 dropped -> 1, 2, 4, 5)
    df_gap = df_valid[~((df_valid["unit_id"] == 1) & (df_valid["cycle"] == 3))].copy()
    with pytest.raises(ValidationError, match="invalid cycle step"):
        validate_trajectory_chronology(df_gap, "train")
        
    # Duplicate cycle
    df_dup = df_valid.copy()
    df_dup = pd.concat([df_dup, df_dup.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValidationError, match="duplicate.*keys"):
        validate_trajectory_chronology(df_dup, "train")


def test_validate_finite_values():
    """Test rejection of NaNs and infinite values."""
    df = create_synthetic_cmapss_df(n_units=1, cycles_per_unit=5)
    validate_finite_values(df, "train")
    
    # With NaN
    df_nan = df.copy()
    df_nan.loc[0, "s02"] = np.nan
    with pytest.raises(ValidationError, match="Null values detected"):
        validate_finite_values(df_nan, "train")
        
    # With Inf
    df_inf = df.copy()
    df_inf.loc[0, "s03"] = np.inf
    with pytest.raises(ValidationError, match="Non-finite values"):
        validate_finite_values(df_inf, "train")


def test_detect_constant_columns():
    """Test detection of constant columns."""
    df = create_synthetic_cmapss_df(n_units=2, cycles_per_unit=10, constant_sensors=["s01", "s05", "s16"])
    constant_cols, nonconstant_cols = detect_constant_columns(df, SENSOR_COLUMNS)
    
    assert set(constant_cols) == {"s01", "s05", "s16"}
    assert len(nonconstant_cols) == 18
    assert "s02" in nonconstant_cols


def test_run_data_quality_checks():
    """Test full quality check suite generating QualityReport."""
    train_df = create_synthetic_cmapss_df(n_units=3, cycles_per_unit=25, constant_sensors=["s01", "s05"])
    test_df = create_synthetic_cmapss_df(n_units=2, cycles_per_unit=15, constant_sensors=["s01", "s05"])
    test_df["source_partition"] = "test"
    rul_array = np.array([50.0, 75.0])
    
    report = run_data_quality_checks(train_df, test_df, rul_array)
    assert report.dataset_id == "FD001"
    assert report.train_rows == 75
    assert report.train_engines == 3
    assert report.test_rows == 30
    assert report.test_engines == 2
    assert report.chronology_valid is True
    assert "s01" in report.constant_sensors_train
    assert "s05" in report.constant_sensors_train


def test_real_fd001_raw_data_validation():
    """Validate real downloaded FD001 data files if present."""
    raw_dir = Path("data/raw/FD001")
    train_file = raw_dir / "train_FD001.txt"
    test_file = raw_dir / "test_FD001.txt"
    rul_file = raw_dir / "RUL_FD001.txt"
    
    if not (train_file.exists() and test_file.exists() and rul_file.exists()):
        pytest.skip("FD001 raw data not present locally.")
        
    train_df = load_raw_cmapss_file(train_file, "train")
    test_df = load_raw_cmapss_file(test_file, "test")
    rul_array = load_raw_rul_file(rul_file)
    
    report = run_data_quality_checks(train_df, test_df, rul_array)
    assert report.train_rows == 20631
    assert report.train_engines == 100
    assert report.test_rows == 13096
    assert report.test_engines == 100
    assert report.test_rul_count == 100
    assert report.columns_count == 26
    assert report.chronology_valid is True
    assert report.has_nulls is False
    assert report.has_infinite is False
