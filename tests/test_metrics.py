"""Unit tests for evaluation metrics and NASA asymmetric scoring."""

import numpy as np

from turbineguard.evaluate import compute_nasa_asymmetric_score, compute_rul_metrics


def test_nasa_asymmetric_score_values_and_asymmetry():
    """
    Test NASA asymmetric scoring function:
    - Zero error gives zero score.
    - Error +10 (overestimation) is penalised more than Error -10 (underestimation).
    """
    # e = 0
    assert np.isclose(compute_nasa_asymmetric_score(np.array([0.0]))[0], 0.0)
    
    # e = +10: exp(10/10) - 1 = e - 1 ~ 1.7182818
    score_pos10 = compute_nasa_asymmetric_score(np.array([10.0]))[0]
    expected_pos10 = np.e - 1.0
    assert np.isclose(score_pos10, expected_pos10)
    
    # e = -10: exp(10/13) - 1 ~ 1.1581825
    score_neg10 = compute_nasa_asymmetric_score(np.array([-10.0]))[0]
    expected_neg10 = np.exp(10.0 / 13.0) - 1.0
    assert np.isclose(score_neg10, expected_neg10)
    
    # Verification of asymmetry
    assert score_pos10 > score_neg10


def test_compute_rul_metrics_hand_calculated():
    """Test MAE, RMSE, bias, overestimation rates on known numbers."""
    y_true = np.array([10.0, 50.0, 120.0, 20.0])
    y_pred = np.array([15.0, 40.0, 140.0, 20.0])
    # Errors: [+5, -10, +20, 0]
    # abs(errors): [5, 10, 20, 0] -> MAE = 35/4 = 8.75
    # sq(errors): [25, 100, 400, 0] -> RMSE = sqrt(525/4) = sqrt(131.25) ~ 11.4564
    # bias: (5 - 10 + 20 + 0)/4 = 15/4 = 3.75
    # overestimation (>0): +5, +20 (2/4 = 0.5)
    # overestimation >10: +20 (1/4 = 0.25)
    
    metrics = compute_rul_metrics(y_pred, y_true, clip_zero=True)
    assert metrics.n_samples == 4
    assert np.isclose(metrics.mae, 8.75)
    assert np.isclose(metrics.rmse, np.sqrt(131.25))
    assert np.isclose(metrics.bias, 3.75)
    assert np.isclose(metrics.overestimate_rate, 0.5)
    assert np.isclose(metrics.overestimate_gt_10_rate, 0.25)
    
    # Bands check:
    # y_true <= 30: y_true=[10, 20], errors=[+5, 0] -> mae=2.5, count=2
    assert metrics.band_metrics["rul_le_30"]["n_samples"] == 2
    assert np.isclose(metrics.band_metrics["rul_le_30"]["mae"], 2.5)
    
    # y_true in (30, 100]: y_true=[50], error=[-10] -> mae=10.0, count=1
    assert metrics.band_metrics["rul_31_to_100"]["n_samples"] == 1
    assert np.isclose(metrics.band_metrics["rul_31_to_100"]["mae"], 10.0)
    
    # y_true > 100: y_true=[120], error=[+20] -> mae=20.0, count=1
    assert metrics.band_metrics["rul_gt_100"]["n_samples"] == 1
    assert np.isclose(metrics.band_metrics["rul_gt_100"]["mae"], 20.0)


def test_compute_rul_metrics_zero_clipping():
    """Verify negative raw predictions are clipped to 0 before evaluating against true RUL."""
    y_true = np.array([5.0])
    y_pred_neg = np.array([-10.0])
    
    # With clipping (default): pred becomes 0 -> error is 0 - 5 = -5 -> MAE = 5
    m_clipped = compute_rul_metrics(y_pred_neg, y_true, clip_zero=True)
    assert np.isclose(m_clipped.mae, 5.0)
    
    # Without clipping: error is -10 - 5 = -15 -> MAE = 15
    m_unclipped = compute_rul_metrics(y_pred_neg, y_true, clip_zero=False)
    assert np.isclose(m_unclipped.mae, 15.0)
