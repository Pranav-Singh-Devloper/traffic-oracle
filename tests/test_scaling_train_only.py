import numpy as np

from traffic.data.missing import impute_inputs, tod_means
from traffic.data.scaling import MaskedStandardScaler
from traffic.data.split import chronological_split


def test_scaler_ignores_zeros_and_later_splits(synthetic_metr):
    x = synthetic_metr.to_numpy()
    m = x > 0
    sp = chronological_split(len(x))
    a = MaskedStandardScaler().fit(x[sp.train], m[sp.train])
    x2 = x.copy()
    x2[sp.val.start :] = x2[sp.val.start :] * 10 + 5
    b = MaskedStandardScaler().fit(x2[sp.train], m[sp.train])
    assert a.mean == b.mean and a.std == b.std
    assert abs(a.mean - x[sp.train][m[sp.train]].mean()) < 1e-9


def test_tod_means_and_imputation_train_only(synthetic_metr):
    x = synthetic_metr.to_numpy()
    m = x > 0
    idx = synthetic_metr.index
    sp = chronological_split(len(x))
    t1 = tod_means(x, m, idx, sp.train)
    x2 = x.copy()
    x2[sp.val.start :] += 100
    t2 = tod_means(x2, m, idx, sp.train)
    # allclose: numpy summation order varies with memory alignment (~1e-14 noise)
    np.testing.assert_allclose(t1, t2, rtol=1e-12, atol=1e-12)


def test_per_sensor_and_roundtrip(synthetic_metr):
    x = synthetic_metr.to_numpy()
    s = MaskedStandardScaler().fit(x, x > 0, per_sensor=True)
    np.testing.assert_allclose(s.inverse_transform(s.transform(x)), x)


def test_impute_fills_everything_and_respects_ffill(synthetic_metr):
    x = synthetic_metr.to_numpy()
    m = x > 0
    idx = synthetic_metr.index
    tm = tod_means(x, m, idx, slice(0, len(x)))
    out = impute_inputs(x, m, idx, tm)
    assert (out > 0).all()
    np.testing.assert_array_equal(out[m], x[m])  # observed values untouched
    np.testing.assert_array_equal(impute_inputs(x, m, idx, tm, method="zero_mask"), x)
