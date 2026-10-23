import numpy as np
import pytest

from traffic.evaluation.metrics import masked_mae, masked_mape, masked_rmse
from traffic.evaluation.stats_tests import block_bootstrap_ci, diebold_mariano

Y = np.array([[10.0, 20.0], [30.0, 40.0]])
P = np.array([[12.0, 20.0], [27.0, 0.0]])
M = np.array([[True, True], [True, False]])  # last entry masked, huge error ignored


def test_mae_rmse_mape_hand_computed():
    # errors on observed entries: 2, 0, 3
    assert masked_mae(Y, P, M) == pytest.approx(5 / 3)
    assert masked_rmse(Y, P, M) == pytest.approx(np.sqrt((4 + 0 + 9) / 3))
    assert masked_mape(Y, P, M) == pytest.approx((0.2 + 0.0 + 0.1) / 3 * 100)


def test_all_masked_is_nan():
    with pytest.warns(UserWarning):
        assert np.isnan(masked_mae(Y, P, np.zeros_like(M)))
        assert np.isnan(masked_rmse(Y, P, np.zeros_like(M)))
        assert np.isnan(masked_mape(Y, P, np.zeros_like(M)))


def test_mape_excludes_tiny_targets():
    y = np.array([0.5, 10.0])
    p = np.array([5.0, 11.0])
    assert masked_mape(y, p, np.ones(2, bool)) == pytest.approx(10.0)


def test_nan_predictions_are_ignored():
    p = np.array([[np.nan, 20.0], [27.0, 0.0]])
    assert masked_mae(Y, p, M) == pytest.approx((0 + 3) / 2)


def test_dm_identical_and_different():
    rng = np.random.default_rng(0)
    y = rng.uniform(20, 60, (400, 5))
    m = np.ones_like(y, bool)
    a = y + rng.normal(0, 1, y.shape)
    stat, p = diebold_mariano(y, a, a, m, h=3)
    assert stat == 0.0 and p == 1.0
    b = y + rng.normal(0, 3, y.shape)
    stat, p = diebold_mariano(y, a, b, m, h=3)
    assert stat < 0 and p < 0.01  # model a is clearly better


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    err, cnt = rng.uniform(5, 15, 20), np.full(20, 10.0)
    lo, hi = block_bootstrap_ci(err, cnt, n_boot=500)
    assert lo <= err.sum() / cnt.sum() <= hi
