"""Configuration schema and loader for TurbineGuard."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "TurbineGuard"
    version: str = "0.1.0"
    seed: int = 42


class DataConfig(BaseModel):
    dataset_id: str = "FD001"
    raw_dir: str = "data/raw/FD001"
    processed_dir: str = "data/processed"
    source_manifest: str = "data/source_manifest.json"
    quality_report: str = "reports/data_quality.json"
    split_manifest: str = "data/processed/split_manifest.json"
    snapshot_manifest: str = "data/processed/snapshot_manifest.json"
    features_dev: str = "data/processed/features_dev.parquet"
    features_val: str = "data/processed/features_val.parquet"
    features_test: str = "data/processed/features_test.parquet"


class SplitsConfig(BaseModel):
    dev_engines_count: int = 80
    val_engines_count: int = 20
    n_folds: int = 5
    seed: int = 42
    val_snapshot_offsets: list[int] = Field(default_factory=lambda: [10, 30, 60, 90])
    min_history_cycles: int = 20


class FeaturesConfig(BaseModel):
    window_size: int = 20
    min_history: int = 20
    target_cap: int = 125
    high_rul_proxy_threshold: int = 100
    near_failure_proxy_threshold: int = 30


class PolicyConfig(BaseModel):
    horizon_cycles: int = 30
    capacity_fraction: float = 0.20
    anomaly_percentile: float = 99.0


class ModelsConfig(BaseModel):
    staging_dir: str = "models/staging"
    tracking_uri: str = "sqlite:///mlruns.db"


class Config(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)


def load_config(config_path: str | Path = "configs/default.yaml") -> Config:
    """Load and validate configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}
        
    return Config(**raw_dict)
