"""FastAPI application for TurbineGuard predictive maintenance inference."""

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.schemas import (
    BatchPredictResponse,
    ContributionItem,
    EngineExplanation,
    ExplanationDetail,
    FeatureAttribution,
    HealthResponse,
    LegacyPredictRequest,
    ModelInfoResponse,
    ReadyResponse,
    SingleEnginePredictRequest,
    SingleEnginePredictResponse,
    WorklistItem,
)
from turbineguard.artifacts import LoadedBundle, load_model_bundle
from turbineguard.explain import explain_batch_features, explain_prediction_sample
from turbineguard.features import extract_engine_features_at_cycle
from turbineguard.policy import compute_inspection_priority, is_within_horizon
from turbineguard.predict import score_batch_history

# Global state container for loaded bundle
bundle_state: dict[str, LoadedBundle | None] = {"bundle": None}

MAX_BODY_SIZE_BYTES = 1024 * 1024  # 1 MiB


class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing a maximum 1 MiB request body size."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_BODY_SIZE_BYTES:
                    # Support modern Starlette HTTP_413_CONTENT_TOO_LARGE or 413
                    status_code = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
                    return JSONResponse(
                        status_code=status_code,
                        content={"detail": f"Request body size exceeds 1 MiB limit ({content_length} bytes)."},
                    )
            except ValueError:
                pass
        return await call_next(request)


def get_default_bundle_path() -> Path:
    bundle_env = os.getenv("TURBINEGUARD_BUNDLE_DIR")
    if bundle_env:
        return Path(bundle_env)
    if Path("models/v0.2.1").exists():
        return Path("models/v0.2.1")
    if Path("models/v0.2.0").exists():
        return Path("models/v0.2.0")
    return Path("models/v0.1.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: attempt to load model bundle if available
    bundle_path = get_default_bundle_path()
    if bundle_path.exists():
        try:
            bundle_state["bundle"] = load_model_bundle(bundle_path)
            app.state.bundle = bundle_state["bundle"]
        except (FileNotFoundError, ValueError, KeyError, RuntimeError):
            # Incomplete or missing bundle
            bundle_state["bundle"] = None
            app.state.bundle = None
    else:
        bundle_state["bundle"] = None
        app.state.bundle = None
    yield
    # Shutdown
    bundle_state["bundle"] = None
    app.state.bundle = None


app = FastAPI(
    title="TurbineGuard Predictive Maintenance API",
    description="Operational RUL estimation, anomaly detection, and inspection ranking service.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(LimitUploadSizeMiddleware)


def get_bundle() -> LoadedBundle:
    """Retrieve active loaded model bundle or attempt lazy loading. Raise 503 if unavailable."""
    bundle = bundle_state.get("bundle")
    if bundle is None:
        bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        bundle_path = get_default_bundle_path()
        if bundle_path.exists():
            try:
                bundle = load_model_bundle(bundle_path)
                bundle_state["bundle"] = bundle
                app.state.bundle = bundle
                return bundle
            except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Failed to load model bundle: {e!s}",
                ) from e
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle is not loaded or bundle directory does not exist.",
        )
    return bundle


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service liveness probe (returns 200 even if model is not loaded)."""
    bundle = bundle_state.get("bundle") or getattr(app.state, "bundle", None)
    return HealthResponse(
        status="healthy",
        bundle_version=bundle.version if bundle else None,
        timestamp_utc=datetime.now(UTC).isoformat(),
    )


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Service readiness probe (returns 200 only if model bundle is loaded, 503 otherwise)."""
    bundle = get_bundle()
    return ReadyResponse(
        status="ready",
        model_loaded=True,
        bundle_version=bundle.version,
        promotion_status=bundle.metadata.get("promotion_status", "unknown"),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Get metadata, policy configuration, and validation gates for active model."""
    bundle = get_bundle()
    meta = bundle.metadata
    return ModelInfoResponse(
        bundle_version=bundle.version,
        dataset_id=meta.get("dataset_id", "FD001"),
        champion_experiment_id=meta.get("champion_experiment_id", "E05"),
        model_family=meta.get("model_family", "xgboost"),
        promotion_status=meta.get("promotion_status", "promoted"),
        policy=bundle.policy,
        feature_schema=bundle.feature_schema,
        validation_gates=meta.get("validation_gates", {}),
        file_hashes=meta.get("file_hashes", {}),
    )


@app.post("/predict")
async def predict(request: Request) -> Any:
    """
    Unified prediction endpoint supporting:
    1. PRD Single-engine history scoring (SingleEnginePredictRequest) -> SingleEnginePredictResponse
    2. Legacy batch sensor readings (LegacyPredictRequest) -> BatchPredictResponse
    """
    bundle = get_bundle()

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON payload: {e!s}",
        ) from e

    if not isinstance(body, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request body must be a JSON object.",
        )

    # 1. Check if PRD Single-Engine History schema
    if "history" in body:
        try:
            req = SingleEnginePredictRequest(**body)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Validation error for single-engine history: {e!s}",
            ) from e

        # Convert history rows to DataFrame
        history_dicts = [r.model_dump() for r in req.history]
        df_history = pd.DataFrame(history_dicts)
        df_history["unit_id"] = 1  # internal engine dummy key for extraction
        df_history["source_partition"] = "predict"

        latest_c = int(df_history["cycle"].max())
        history_len = len(df_history)

        try:
            feats_dict = extract_engine_features_at_cycle(
                df_history,
                target_cycle=latest_c,
                window_size=bundle.feature_schema.get("window_size", 20),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Feature extraction failed on history: {e!s}",
            ) from e

        df_feat = pd.DataFrame([feats_dict])

        # RUL Score
        rul_features = bundle.feature_schema["rul_feature_names"]
        raw_rul = float(bundle.rul_pipeline.predict(df_feat[rul_features])[0])
        est_rul = max(0.0, raw_rul)
        clipping_applied = bool(raw_rul < 0.0)

        # Anomaly Score
        ano_features = bundle.feature_schema["anomaly_feature_names"]
        cutoff = float(bundle.policy["anomaly_cutoff"])
        from turbineguard.anomaly import score_anomaly_samples
        ano_scores, ano_flags = score_anomaly_samples(bundle.anomaly_pipeline, df_feat, ano_features, cutoff)
        ano_score = float(ano_scores[0])
        ano_flag = bool(ano_flags[0])

        horizon = int(bundle.policy.get("horizon_cycles", 30))
        in_horizon = is_within_horizon(est_rul, horizon)
        priority = compute_inspection_priority(in_horizon, ano_flag)

        explanation = None
        if req.explain:
            exp_res = explain_prediction_sample(bundle, df_feat.iloc[0], top_k=3)
            contribs = [
                ContributionItem(
                    feature=t["feature"],
                    feature_value=float(t["feature_value"]),
                    contribution_cycles=float(t["shap_value"]),
                    effect=t["effect"],
                )
                for t in exp_res.get("top_features", [])
            ]
            explanation = ExplanationDetail(
                raw_output=raw_rul,
                base_value=float(exp_res.get("base_value", 0.0)),
                clipping_applied=clipping_applied,
                top_contributions=contribs,
            )

        return SingleEnginePredictResponse(
            engine_id=req.engine_id,
            dataset_id=req.dataset_id,
            latest_cycle=latest_c,
            history_length=history_len,
            estimated_rul_cycles=est_rul,
            within_horizon=in_horizon,
            anomaly_score=ano_score,
            anomaly_cutoff=cutoff,
            anomaly_flag=ano_flag,
            inspection_priority=priority,
            bundle_version=bundle.version,
            policy_version=bundle.policy.get("policy_version", "1.0"),
            schema_version=bundle.feature_schema.get("schema_version", "1.0"),
            explanation=explanation,
        )

    # 2. Check legacy readings schema
    elif "readings" in body:
        try:
            req_legacy = LegacyPredictRequest(**body)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Validation error for batch readings: {e!s}",
            ) from e

        data_dicts = [r.model_dump() for r in req_legacy.readings]
        df_input = pd.DataFrame(data_dicts)

        try:
            ranked_worklist, latest_features = score_batch_history(bundle, df_input)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Batch inference failed: {e!s}",
            ) from e

        worklist_items = [
            WorklistItem(
                unit_id=int(row["unit_id"]),
                latest_cycle=int(row["latest_cycle"]),
                estimated_rul=float(row["estimated_rul"]),
                within_horizon=bool(row["within_horizon"]),
                anomaly_score=float(row["anomaly_score"]),
                anomaly_flag=bool(row["anomaly_flag"]),
                priority=str(row["priority"]),
                rank=int(row["rank"]),
            )
            for _, row in ranked_worklist.iterrows()
        ]

        explanations = None
        if req_legacy.explain:
            raw_exps = explain_batch_features(bundle, latest_features, top_k=5)
            explanations = [
                EngineExplanation(
                    unit_id=exp["unit_id"],
                    latest_cycle=exp["latest_cycle"],
                    base_value=exp["base_value"],
                    estimated_rul=exp["estimated_rul"],
                    top_features=[
                        FeatureAttribution(
                            feature=f["feature"],
                            feature_value=f["feature_value"],
                            shap_value=f["shap_value"],
                            effect=f["effect"],
                        )
                        for f in exp["top_features"]
                    ],
                )
                for exp in raw_exps
            ]

        return BatchPredictResponse(
            bundle_version=bundle.version,
            ranked_worklist=worklist_items,
            explanations=explanations,
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request must contain either 'history' (single-engine PRD format) or 'readings' (batch format).",
        )
