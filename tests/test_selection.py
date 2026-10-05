"""Unit tests for champion model selection and 1-SE rule."""

from turbineguard.train import ExperimentResult, FoldEvaluationResult, select_champion_model


def make_dummy_result(exp_id: str, family: str, mean_rmse: float, std_err: float, best_params: dict | None = None) -> ExperimentResult:
    """Helper to create dummy ExperimentResult."""
    fold_res = [
        FoldEvaluationResult(fold_idx=i, train_engines_count=64, val_engines_count=16, rmse=mean_rmse, mae=mean_rmse, bias=0.0, nasa_score_mean=1.0)
        for i in range(5)
    ]
    return ExperimentResult(
        experiment_id=exp_id,
        model_family=family,
        description="test",
        feature_set=["f1"],
        best_params=best_params or {},
        fold_results=fold_res,
        mean_cv_rmse=mean_rmse,
        std_err_cv_rmse=std_err,
    )


def test_select_champion_picks_lowest_rmse():
    """Verify champion selection picks best model when no other model is within 1 SE."""
    e02 = make_dummy_result("E02", "baseline", mean_rmse=29.0, std_err=1.0)
    e03 = make_dummy_result("E03", "baseline", mean_rmse=18.0, std_err=1.0)
    e05 = make_dummy_result("E05", "xgboost", mean_rmse=12.0, std_err=0.5, best_params={"max_depth": 3, "n_estimators": 100})
    
    # E05 is best at 12.0 (tolerance is 12.5). E03 at 18.0 is outside tolerance.
    champ, _ = select_champion_model([e02, e03, e05])
    assert champ.experiment_id == "E05"


def test_select_champion_prefers_ridge_within_1se():
    """Verify champion selection prefers simpler Ridge when within 1 SE of best XGBoost."""
    e03 = make_dummy_result("E03", "baseline", mean_rmse=14.5, std_err=1.0, best_params={"alpha": 10.0})
    e05 = make_dummy_result("E05", "xgboost", mean_rmse=14.0, std_err=1.0, best_params={"max_depth": 4, "n_estimators": 200})
    
    # Best is E05 at 14.0, SE is 1.0 -> tolerance is 15.0. E03 at 14.5 is within tolerance.
    champ, reason = select_champion_model([e03, e05])
    assert champ.experiment_id == "E03"
    assert "Ridge candidate" in reason


def test_select_champion_prefers_shallower_xgboost_within_1se():
    """Verify champion selection prefers shallower/smaller XGBoost when within 1 SE."""
    e05_deep = make_dummy_result("E05", "xgboost", mean_rmse=13.0, std_err=1.0, best_params={"max_depth": 5, "n_estimators": 200})
    e04_shallow = make_dummy_result("E04", "xgboost", mean_rmse=13.5, std_err=1.0, best_params={"max_depth": 2, "n_estimators": 100})
    
    # Best is E05_deep (13.0, tol 14.0). E04_shallow (13.5, depth=2) should be preferred over depth=5.
    champ, reason = select_champion_model([e05_deep, e04_shallow])
    assert champ.experiment_id == "E04"
    assert "depth=2" in reason
