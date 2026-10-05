"""Pydantic request and response schemas for TurbineGuard REST API."""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = "healthy"
    bundle_version: str | None = None
    timestamp_utc: str


class ReadyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = "ready"
    model_loaded: bool = True
    bundle_version: str
    promotion_status: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bundle_version: str
    dataset_id: str
    champion_experiment_id: str
    model_family: str
    promotion_status: str
    policy: dict[str, Any]
    feature_schema: dict[str, Any]
    validation_gates: dict[str, Any]
    file_hashes: dict[str, str]


class HistoryCycleReading(BaseModel):
    """Single cycle reading for an engine history."""
    model_config = ConfigDict(extra="forbid")

    cycle: int = Field(..., ge=1, description="Positive integer cycle index")
    op_1: float = Field(..., description="Operational setting 1")
    op_2: float = Field(..., description="Operational setting 2")
    op_3: float = Field(..., description="Operational setting 3")
    s01: float
    s02: float
    s03: float
    s04: float
    s05: float
    s06: float
    s07: float
    s08: float
    s09: float
    s10: float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s16: float
    s17: float
    s18: float
    s19: float
    s20: float
    s21: float

    @field_validator("*", mode="after")
    @classmethod
    def check_finite(cls, v: Any) -> Any:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("Values must be finite (NaN and Inf are not allowed).")
        return v


class SingleEnginePredictRequest(BaseModel):
    """PRD single-engine history scoring request contract."""
    model_config = ConfigDict(extra="forbid")

    dataset_id: Literal["FD001"] = Field(default="FD001", description="Dataset identifier, strictly FD001")
    engine_id: str = Field(..., min_length=1, max_length=64, description="Nonempty engine identifier")
    history: list[HistoryCycleReading] = Field(
        ...,
        min_length=20,
        max_length=200,
        description="Consecutive ascending sensor history (20 to 200 cycles)",
    )
    explain: bool = Field(default=False, description="Whether to compute top-3 SHAP feature explanations")

    @field_validator("history")
    @classmethod
    def validate_chronology(cls, history: list[HistoryCycleReading]) -> list[HistoryCycleReading]:
        cycles = [r.cycle for r in history]
        # Check ascending and consecutive
        for i in range(1, len(cycles)):
            if cycles[i] != cycles[i - 1] + 1:
                raise ValueError(
                    f"History cycles must be strictly ascending and consecutive without gaps. "
                    f"Found step from cycle {cycles[i-1]} to {cycles[i]}."
                )
        return history


class LegacySensorReading(BaseModel):
    model_config = ConfigDict(extra="ignore")
    unit_id: int
    cycle: int = Field(..., ge=1)
    op_1: float
    op_2: float
    op_3: float
    s01: float
    s02: float
    s03: float
    s04: float
    s05: float
    s06: float
    s07: float
    s08: float
    s09: float
    s10: float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s16: float
    s17: float
    s18: float
    s19: float
    s20: float
    s21: float
    source_partition: str | None = None


class LegacyPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    readings: list[LegacySensorReading] = Field(..., min_length=20)
    explain: bool = Field(default=False)


class ContributionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature: str
    feature_value: float
    contribution_cycles: float
    effect: str


class ExplanationDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_output: float
    base_value: float
    clipping_applied: bool
    top_contributions: list[ContributionItem]


class SingleEnginePredictResponse(BaseModel):
    """PRD single-engine prediction response contract."""
    model_config = ConfigDict(extra="forbid")

    engine_id: str
    dataset_id: str
    latest_cycle: int
    history_length: int
    estimated_rul_cycles: float
    within_horizon: bool
    anomaly_score: float
    anomaly_cutoff: float
    anomaly_flag: bool
    inspection_priority: str
    bundle_version: str
    policy_version: str
    schema_version: str
    explanation: ExplanationDetail | None = None


# Backward-compatible batch schema
class WorklistItem(BaseModel):
    unit_id: int
    latest_cycle: int
    estimated_rul: float
    within_horizon: bool
    anomaly_score: float
    anomaly_flag: bool
    priority: str
    rank: int


class FeatureAttribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    effect: str


class EngineExplanation(BaseModel):
    unit_id: int
    latest_cycle: int | None
    base_value: float
    estimated_rul: float
    top_features: list[FeatureAttribution]


class BatchPredictResponse(BaseModel):
    bundle_version: str
    ranked_worklist: list[WorklistItem]
    explanations: list[EngineExplanation] | None = None
