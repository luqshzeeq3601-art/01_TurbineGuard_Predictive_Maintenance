"""Unit and regression tests for SHAP model explanations."""

from turbineguard.explain import explain_batch_features, explain_prediction_sample
from turbineguard.predict import score_batch_history


def test_single_engine_shap_explanation(loaded_bundle, synthetic_history):
    # Load test data and score
    test_df = synthetic_history
    engine_1 = test_df[test_df["unit_id"] == 1]
    
    _, latest_feats = score_batch_history(loaded_bundle, engine_1)
    exp = explain_prediction_sample(loaded_bundle, latest_feats.iloc[0], top_k=5)

    assert exp["unit_id"] == 1
    assert exp["latest_cycle"] == 31
    assert isinstance(exp["estimated_rul"], float)
    assert isinstance(exp["base_value"], float)
    assert len(exp["top_features"]) == 5

    # Check structure of feature attributions
    for feat in exp["top_features"]:
        assert "feature" in feat
        assert "feature_value" in feat
        assert "shap_value" in feat
        assert feat["effect"] in ["increases_estimated_rul", "decreases_estimated_rul"]


def test_batch_shap_explanations(loaded_bundle, synthetic_history):
    test_df = synthetic_history
    first_5_engines = test_df[test_df["unit_id"].isin([1, 2, 3, 4, 5])]
    
    _, latest_feats = score_batch_history(loaded_bundle, first_5_engines)
    exps = explain_batch_features(loaded_bundle, latest_feats, top_k=3)

    assert len(exps) == 5
    for exp in exps:
        assert len(exp["top_features"]) == 3
        assert exp["estimated_rul"] >= 0.0
