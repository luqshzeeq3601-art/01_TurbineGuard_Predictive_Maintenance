"""Model training, grouped cross-validation, hyperparameter tuning, and champion selection."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from turbineguard.config import Config, load_config
from turbineguard.evaluate import compute_rul_metrics

logger = logging.getLogger(__name__)


@dataclass
class FoldEvaluationResult:
    fold_idx: int
    train_engines_count: int
    val_engines_count: int
    rmse: float
    mae: float
    bias: float
    nasa_score_mean: float


@dataclass
class ExperimentResult:
    experiment_id: str
    model_family: str
    description: str
    feature_set: list[str]
    best_params: dict[str, Any]
    fold_results: list[FoldEvaluationResult]
    mean_cv_rmse: float
    std_err_cv_rmse: float
    validation_rmse: float | None = None
    validation_metrics: dict | None = None


def get_feature_columns_for_experiment(
    experiment_id: str,
    df_columns: list[str],
    constant_sensors: list[str],
) -> list[str]:
    """Determine input feature column names based on experiment ID."""
    if experiment_id in ["E01", "E02"]:
        return ["cycle"]
    elif experiment_id in ["E03", "E04"]:
        sensor_last_cols = [
            f"s{i:02d}_last"
            for i in range(1, 22)
            if f"s{i:02d}" not in constant_sensors and f"s{i:02d}_last" in df_columns
        ]
        return ["cycle", "op_1", "op_2", "op_3"] + sensor_last_cols
    elif experiment_id in ["E05", "E07"]:
        feature_cols = ["cycle", "op_1", "op_2", "op_3"]
        for i in range(1, 22):
            s_name = f"s{i:02d}"
            if s_name not in constant_sensors:
                for suffix in ["last", "mean", "std", "slope"]:
                    col = f"{s_name}_{suffix}"
                    if col in df_columns:
                        feature_cols.append(col)
        return feature_cols
    elif experiment_id == "E06":
        # Winner without cycle
        feature_cols = ["op_1", "op_2", "op_3"]
        for i in range(1, 22):
            s_name = f"s{i:02d}"
            if s_name not in constant_sensors:
                for suffix in ["last", "mean", "std", "slope"]:
                    col = f"{s_name}_{suffix}"
                    if col in df_columns:
                        feature_cols.append(col)
        return feature_cols
    else:
        raise ValueError(f"Unknown experiment ID: {experiment_id}")


def evaluate_model_on_heldout_engines(
    model: Any,
    feature_cols: list[str],
    heldout_df: pd.DataFrame,
    eval_offsets: list[int] | None = None,
) -> tuple[float, float, float, float]:
    """
    Evaluate trained model on held-out development engines:
    - For each engine, evaluate snapshots corresponding to eligible offsets in [10, 30, 60, 90].
    - Predict RUL, clip at 0.
    - True RUL is the offset.
    - Average squared errors per engine (equal engine weights).
    - Return (rmse, mae, bias, nasa_score_mean).
    """
    eval_offsets = eval_offsets or [10, 30, 60, 90]
    engine_sq_errors = []
    all_preds = []
    all_trues = []

    for unit_id, engine_rows in heldout_df.groupby("unit_id"):
        max_c = engine_rows["cycle"].max()
        e_sq_errs = []

        for offset in eval_offsets:
            target_c = max_c - offset
            snapshot_row = engine_rows[engine_rows["cycle"] == target_c]
            if len(snapshot_row) == 0:
                continue

            X_snap = snapshot_row[feature_cols]
            pred_raw = model.predict(X_snap)[0]
            pred_clipped = max(0.0, float(pred_raw))
            true_rul = float(offset)

            err = pred_clipped - true_rul
            e_sq_errs.append(err**2)
            all_preds.append(pred_clipped)
            all_trues.append(true_rul)

        if e_sq_errs:
            engine_sq_errors.append(np.mean(e_sq_errs))

    if not engine_sq_errors:
        raise ValueError("No eligible snapshots found in held-out data.")

    fold_mse = float(np.mean(engine_sq_errors))
    fold_rmse = float(np.sqrt(fold_mse))

    metrics = compute_rul_metrics(all_preds, all_trues, clip_zero=True)
    return fold_rmse, metrics.mae, metrics.bias, metrics.nasa_score_mean


def train_and_evaluate_baseline_cv(
    experiment_id: str,
    dev_df: pd.DataFrame,
    split_manifest: dict,
    cfg: Config,
) -> ExperimentResult:
    """Run 5-fold grouped CV for a baseline model (E01, E02, or E03)."""
    cv_folds = split_manifest["cv_folds"]
    alpha_candidates = [0.1, 1.0, 10.0, 100.0] if experiment_id in ["E02", "E03"] else [None]

    best_overall_rmse = float("inf")
    best_param = None
    best_fold_results = []

    for alpha in alpha_candidates:
        fold_results = []
        fold_rmses = []

        for fold_idx, (fold_name, val_engine_ids) in enumerate(cv_folds.items()):
            val_mask = dev_df["unit_id"].isin(val_engine_ids)
            train_mask = ~val_mask

            train_data = dev_df[train_mask].copy()
            val_data = dev_df[val_mask].copy()

            # Determine constant columns on this training fold only
            train_raw_sensors = [f"s{i:02d}" for i in range(1, 22)]
            const_sensors = []
            for s in train_raw_sensors:
                col = f"{s}_last"
                if col in train_data.columns and train_data[col].std(ddof=0) <= 1e-12:
                    const_sensors.append(s)

            feat_cols = get_feature_columns_for_experiment(
                experiment_id,
                train_data.columns.tolist(),
                const_sensors,
            )

            X_train = train_data[feat_cols]
            y_train = train_data["rul_capped"]
            sample_weights = train_data["sample_weight"].values

            if experiment_id == "E01":
                model = DummyRegressor(strategy="median")
                model.fit(X_train, y_train, sample_weight=sample_weights)
            elif experiment_id in ["E02", "E03"]:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(alpha=alpha, random_state=cfg.project.seed)),
                ])
                pipe.fit(X_train, y_train, regressor__sample_weight=sample_weights)
                model = pipe

            rmse, mae, bias, nasa_mean = evaluate_model_on_heldout_engines(
                model=model,
                feature_cols=feat_cols,
                heldout_df=val_data,
                eval_offsets=cfg.splits.val_snapshot_offsets,
            )

            fold_rmses.append(rmse)
            fold_results.append(
                FoldEvaluationResult(
                    fold_idx=fold_idx,
                    train_engines_count=int(train_data["unit_id"].nunique()),
                    val_engines_count=len(val_engine_ids),
                    rmse=rmse,
                    mae=mae,
                    bias=bias,
                    nasa_score_mean=nasa_mean,
                )
            )

        mean_rmse = float(np.mean(fold_rmses))
        if mean_rmse < best_overall_rmse:
            best_overall_rmse = mean_rmse
            best_param = {"alpha": alpha} if alpha is not None else {}
            best_fold_results = fold_results

    best_rmses = [r.rmse for r in best_fold_results]
    std_err = float(np.std(best_rmses, ddof=1) / np.sqrt(len(best_rmses)))

    descriptions = {
        "E01": "DummyRegressor median constant baseline",
        "E02": "Ridge age-only baseline (cycle)",
        "E03": "Ridge current-sensors baseline",
    }

    return ExperimentResult(
        experiment_id=experiment_id,
        model_family="baseline",
        description=descriptions.get(experiment_id, ""),
        feature_set=get_feature_columns_for_experiment(experiment_id, dev_df.columns.tolist(), []),
        best_params=best_param or {},
        fold_results=best_fold_results,
        mean_cv_rmse=best_overall_rmse,
        std_err_cv_rmse=std_err,
    )


def run_baseline_experiments(config_path: str | Path = "configs/default.yaml") -> list[ExperimentResult]:
    """Execute E01, E02, and E03 baseline benchmarks with local MLflow tracking."""
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg.models.tracking_uri)
    mlflow.set_experiment("TurbineGuard_Baselines")

    dev_path = Path(cfg.data.features_dev)
    split_path = Path(cfg.data.split_manifest)

    if not (dev_path.exists() and split_path.exists()):
        raise FileNotFoundError("Missing dev features or split manifest. Run prepare first.")

    dev_df = pd.read_parquet(dev_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    results = []
    for exp_id in ["E01", "E02", "E03"]:
        logger.info(f"Running baseline experiment {exp_id}...")
        with mlflow.start_run(run_name=exp_id):
            res = train_and_evaluate_baseline_cv(exp_id, dev_df, split_manifest, cfg)
            results.append(res)

            mlflow.log_params({
                "experiment_id": exp_id,
                "model_family": res.model_family,
                "description": res.description,
                **res.best_params,
            })
            mlflow.log_metrics({
                "mean_cv_rmse": res.mean_cv_rmse,
                "std_err_cv_rmse": res.std_err_cv_rmse,
                **{f"fold_{r.fold_idx}_rmse": r.rmse for r in res.fold_results},
            })

            logger.info(
                f"Completed {exp_id}: Mean CV RMSE = {res.mean_cv_rmse:.2f} "
                f"(± {res.std_err_cv_rmse:.2f}), Best Params = {res.best_params}"
            )

    summary_rows = []
    for r in results:
        summary_rows.append({
            "experiment_id": r.experiment_id,
            "description": r.description,
            "best_params": json.dumps(r.best_params),
            "mean_cv_rmse": r.mean_cv_rmse,
            "std_err_cv_rmse": r.std_err_cv_rmse,
            "fold_0_rmse": r.fold_results[0].rmse,
            "fold_1_rmse": r.fold_results[1].rmse,
            "fold_2_rmse": r.fold_results[2].rmse,
            "fold_3_rmse": r.fold_results[3].rmse,
            "fold_4_rmse": r.fold_results[4].rmse,
        })
    summary_df = pd.DataFrame(summary_rows)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "baseline_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved baseline comparison to {summary_path}")

    return results


def generate_xgboost_search_grid() -> tuple[list[dict], list[dict]]:
    """
    Generate predeclared deterministic search grids for E04 (12 configs) and E05 (18 configs).
    Total = 30 configurations (strictly adheres to <= 30 configuration budget).
    """
    grid_e04 = [
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0},
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.8},
        {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 1.0},
        {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "n_estimators": 150, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.03, "subsample": 1.0, "colsample_bytree": 0.8},
        {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 5, "n_estimators": 150, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 1.0},
    ]

    grid_e05 = [
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0},
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "n_estimators": 150, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.8},
        {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "n_estimators": 150, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.8},
        {"max_depth": 4, "n_estimators": 300, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 5, "n_estimators": 150, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 5, "n_estimators": 200, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
    ]

    return grid_e04, grid_e05


def train_and_evaluate_xgboost_cv(
    experiment_id: str,
    dev_df: pd.DataFrame,
    split_manifest: dict,
    cfg: Config,
    param_grid: list[dict],
) -> tuple[ExperimentResult, list[dict]]:
    """Run grouped CV for XGBoost across candidate parameter configurations."""
    cv_folds = split_manifest["cv_folds"]
    best_overall_rmse = float("inf")
    best_params = None
    best_fold_results = []
    all_config_results = []

    target_col = "rul_true" if experiment_id == "E07" else "rul_capped"

    for config_idx, params in enumerate(param_grid):
        fold_results = []
        fold_rmses = []

        for fold_idx, (fold_name, val_engine_ids) in enumerate(cv_folds.items()):
            val_mask = dev_df["unit_id"].isin(val_engine_ids)
            train_mask = ~val_mask

            train_data = dev_df[train_mask].copy()
            val_data = dev_df[val_mask].copy()

            # Constant sensors on training fold
            train_raw_sensors = [f"s{i:02d}" for i in range(1, 22)]
            const_sensors = []
            for s in train_raw_sensors:
                col = f"{s}_last"
                if col in train_data.columns and train_data[col].std(ddof=0) <= 1e-12:
                    const_sensors.append(s)

            feat_cols = get_feature_columns_for_experiment(
                experiment_id,
                train_data.columns.tolist(),
                const_sensors,
            )

            X_train = train_data[feat_cols]
            y_train = train_data[target_col]
            sample_weights = train_data["sample_weight"].values

            model = xgb.XGBRegressor(
                **params,
                random_state=cfg.project.seed,
                n_jobs=2,
                tree_method="hist",
            )
            model.fit(X_train, y_train, sample_weight=sample_weights)

            rmse, mae, bias, nasa_mean = evaluate_model_on_heldout_engines(
                model=model,
                feature_cols=feat_cols,
                heldout_df=val_data,
                eval_offsets=cfg.splits.val_snapshot_offsets,
            )

            fold_rmses.append(rmse)
            fold_results.append(
                FoldEvaluationResult(
                    fold_idx=fold_idx,
                    train_engines_count=int(train_data["unit_id"].nunique()),
                    val_engines_count=len(val_engine_ids),
                    rmse=rmse,
                    mae=mae,
                    bias=bias,
                    nasa_score_mean=nasa_mean,
                )
            )

        mean_rmse = float(np.mean(fold_rmses))
        std_err = float(np.std(fold_rmses, ddof=1) / np.sqrt(len(fold_rmses)))

        all_config_results.append({
            "experiment_id": experiment_id,
            "config_idx": config_idx,
            "params": params,
            "mean_cv_rmse": mean_rmse,
            "std_err_cv_rmse": std_err,
            "fold_rmses": fold_rmses,
        })

        if mean_rmse < best_overall_rmse:
            best_overall_rmse = mean_rmse
            best_params = params
            best_fold_results = fold_results

    descriptions = {
        "E04": "XGBoost on current values",
        "E05": "XGBoost on current values + 20-cycle rolling features",
        "E06": "Winner without cycle",
        "E07": "Winner trained on uncapped RUL",
    }

    best_rmses = [r.rmse for r in best_fold_results]
    best_std_err = float(np.std(best_rmses, ddof=1) / np.sqrt(len(best_rmses)))

    result = ExperimentResult(
        experiment_id=experiment_id,
        model_family="xgboost",
        description=descriptions.get(experiment_id, ""),
        feature_set=get_feature_columns_for_experiment(experiment_id, dev_df.columns.tolist(), []),
        best_params=best_params or {},
        fold_results=best_fold_results,
        mean_cv_rmse=best_overall_rmse,
        std_err_cv_rmse=best_std_err,
    )
    return result, all_config_results


def select_champion_model(
    results: list[ExperimentResult],
) -> tuple[ExperimentResult, str]:
    """
    Select champion model following docs/05_EXPERIMENT_PLAN.md Section 3:
    1. Pool: nonconstant candidates (E02, E03, E04, E05). E01 is recorded but excluded.
    2. Best model = lowest mean inner-CV endpoint RMSE.
    3. Tolerance = mean_rmse_best + std_err_best.
    4. Preference among models in tolerance:
       - Ridge (E02, E03)
       - Shallower / smaller XGBoost
       - Lowest mean RMSE
    """
    eligible = [r for r in results if r.experiment_id in ["E02", "E03", "E04", "E05"]]
    if not eligible:
        raise ValueError("No eligible models found for champion selection.")

    # Find global lowest mean CV RMSE
    best_candidate = min(eligible, key=lambda x: x.mean_cv_rmse)
    tolerance_threshold = best_candidate.mean_cv_rmse + best_candidate.std_err_cv_rmse

    in_tolerance = [r for r in eligible if r.mean_cv_rmse <= tolerance_threshold]

    # Preference ranking: Ridge first
    ridge_candidates = [r for r in in_tolerance if r.model_family == "baseline" and r.experiment_id != "E01"]
    if ridge_candidates:
        champion = min(ridge_candidates, key=lambda x: x.mean_cv_rmse)
        reason = f"Ridge candidate {champion.experiment_id} within 1 SE ({tolerance_threshold:.2f}) of best ({best_candidate.experiment_id}: {best_candidate.mean_cv_rmse:.2f})"
        return champion, reason

    # Smaller XGBoost (lowest max_depth, then lowest n_estimators)
    xgb_candidates = [r for r in in_tolerance if r.model_family == "xgboost"]
    if xgb_candidates:
        champion = min(
            xgb_candidates,
            key=lambda x: (
                x.best_params.get("max_depth", 99),
                x.best_params.get("n_estimators", 9999),
                x.mean_cv_rmse,
            ),
        )
        reason = f"XGBoost candidate {champion.experiment_id} selected via 1-SE rule (depth={champion.best_params.get('max_depth')}, n_est={champion.best_params.get('n_estimators')})"
        return champion, reason

    return best_candidate, f"Best mean CV RMSE candidate {best_candidate.experiment_id}"


def fit_and_stage_champion(
    champion: ExperimentResult,
    dev_df: pd.DataFrame,
    cfg: Config,
) -> Path:
    """Fit selected champion model on all 80 development engines and stage artifact."""
    staging_dir = Path(cfg.models.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    const_sensors = ["s01", "s05", "s10", "s16", "s18", "s19"]
    feat_cols = get_feature_columns_for_experiment(
        champion.experiment_id,
        dev_df.columns.tolist(),
        const_sensors,
    )

    X_dev = dev_df[feat_cols]
    y_dev = dev_df["rul_capped"]
    weights_dev = dev_df["sample_weight"].values

    if champion.model_family == "baseline":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=champion.best_params.get("alpha", 1.0), random_state=cfg.project.seed)),
        ])
        pipeline.fit(X_dev, y_dev, regressor__sample_weight=weights_dev)
    elif champion.model_family == "xgboost":
        pipeline = xgb.XGBRegressor(
            **champion.best_params,
            random_state=cfg.project.seed,
            n_jobs=2,
            tree_method="hist",
        )
        pipeline.fit(X_dev, y_dev, sample_weight=weights_dev)
    else:
        raise ValueError(f"Unsupported model family: {champion.model_family}")

    model_path = staging_dir / "rul_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    info = {
        "champion_experiment_id": champion.experiment_id,
        "model_family": champion.model_family,
        "description": champion.description,
        "best_params": champion.best_params,
        "feature_names": feat_cols,
        "mean_cv_rmse": champion.mean_cv_rmse,
        "std_err_cv_rmse": champion.std_err_cv_rmse,
    }
    info_path = staging_dir / "champion_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    logger.info(f"Fitted champion {champion.experiment_id} on 80 development engines and staged to {model_path}")
    return model_path


def run_xgboost_experiments(config_path: str | Path = "configs/default.yaml") -> list[ExperimentResult]:
    """Execute E04, E05, champion selection, and diagnostic runs (E06, E07)."""
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg.models.tracking_uri)
    mlflow.set_experiment("TurbineGuard_Modelling")

    dev_path = Path(cfg.data.features_dev)
    split_path = Path(cfg.data.split_manifest)
    dev_df = pd.read_parquet(dev_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    all_results = []
    for base_id in ["E01", "E02", "E03"]:
        res = train_and_evaluate_baseline_cv(base_id, dev_df, split_manifest, cfg)
        all_results.append(res)

    grid_e04, grid_e05 = generate_xgboost_search_grid()

    logger.info("Running E04 XGBoost on current values (12 configurations)...")
    with mlflow.start_run(run_name="E04_XGBoost_Current"):
        res_e04, _ = train_and_evaluate_xgboost_cv("E04", dev_df, split_manifest, cfg, grid_e04)
        all_results.append(res_e04)
        mlflow.log_params({"experiment_id": "E04", **res_e04.best_params})
        mlflow.log_metrics({"mean_cv_rmse": res_e04.mean_cv_rmse, "std_err_cv_rmse": res_e04.std_err_cv_rmse})

    logger.info("Running E05 XGBoost on rolling features (18 configurations)...")
    with mlflow.start_run(run_name="E05_XGBoost_Rolling"):
        res_e05, _ = train_and_evaluate_xgboost_cv("E05", dev_df, split_manifest, cfg, grid_e05)
        all_results.append(res_e05)
        mlflow.log_params({"experiment_id": "E05", **res_e05.best_params})
        mlflow.log_metrics({"mean_cv_rmse": res_e05.mean_cv_rmse, "std_err_cv_rmse": res_e05.std_err_cv_rmse})

    # Select champion
    champion, selection_reason = select_champion_model(all_results)
    logger.info(f"CHAMPION SELECTED: {champion.experiment_id} ({selection_reason})")

    # Fit champion on 80 development engines and stage
    fit_and_stage_champion(champion, dev_df, cfg)

    # Diagnostic E06: Champion without cycle
    logger.info(f"Running diagnostic E06 ({champion.experiment_id} without cycle)...")
    with mlflow.start_run(run_name="E06_Winner_No_Cycle"):
        res_e06, _ = train_and_evaluate_xgboost_cv("E06", dev_df, split_manifest, cfg, [champion.best_params])
        all_results.append(res_e06)

    # Diagnostic E07: Champion trained on uncapped RUL
    logger.info(f"Running diagnostic E07 ({champion.experiment_id} uncapped target)...")
    with mlflow.start_run(run_name="E07_Winner_Uncapped"):
        res_e07, _ = train_and_evaluate_xgboost_cv("E07", dev_df, split_manifest, cfg, [champion.best_params])
        all_results.append(res_e07)

    # Save model comparison table
    comparison_rows = []
    for r in all_results:
        is_champ = (r.experiment_id == champion.experiment_id)
        comparison_rows.append({
            "experiment_id": r.experiment_id,
            "description": r.description,
            "model_family": r.model_family,
            "is_champion": is_champ,
            "selection_notes": selection_reason if is_champ else "",
            "best_params": json.dumps(r.best_params),
            "mean_cv_rmse": r.mean_cv_rmse,
            "std_err_cv_rmse": r.std_err_cv_rmse,
            "fold_0_rmse": r.fold_results[0].rmse,
            "fold_1_rmse": r.fold_results[1].rmse,
            "fold_2_rmse": r.fold_results[2].rmse,
            "fold_3_rmse": r.fold_results[3].rmse,
            "fold_4_rmse": r.fold_results[4].rmse,
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comp_path = Path("reports/model_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    logger.info(f"Saved complete model comparison to {comp_path}")

    return all_results
