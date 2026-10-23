import numpy as np

from traffic.features.lags import lag_features
from traffic.features.spatial import neighbor_features, neighbor_operators


def test_lag_features_ignore_the_future(synthetic_metr):
    x = synthetic_metr.to_numpy().copy()
    m = x > 0
    anchors = np.array([300, 350, 400])
    a = lag_features(x, m, anchors, 12)
    x2 = x.copy()
    x2[401:] = 999.0  # perturb everything after the last anchor
    b = lag_features(x2, m, anchors, 12)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_daily_lag_flag(synthetic_metr):
    x = synthetic_metr.to_numpy()
    m = x > 0
    off = lag_features(x, m, np.array([300]), 12, use_daily_lag=False)
    on = lag_features(x, m, np.array([100, 300]), 12, use_daily_lag=True)
    assert "lag_288" not in off and "lag_288" in on
    assert np.isnan(on["lag_288"][0]).all()  # no data a day before t=100
    np.testing.assert_array_equal(on["lag_288"][1], x[12].astype(np.float32))


def test_spatial_directions_on_chain():
    """Chain a->b->c: W[a,b]>0, W[b,c]>0. b's downstream is c and upstream is a."""
    w = np.zeros((3, 3))
    w[0, 1] = w[1, 2] = 1.0
    ops = neighbor_operators(w)
    np.testing.assert_array_equal(ops["down"][0][1], [0, 0, 1])
    np.testing.assert_array_equal(ops["up"][0][1], [1, 0, 0])
    assert not ops["up"][1][0] and not ops["down"][1][2]  # a has no upstream, c no downstream
    x = np.tile(np.array([10.0, 20.0, 30.0]), (5, 1))
    f = neighbor_features(x, w, np.array([4]))
    assert f["down_mean_lag0"][0, 1] == 30.0 and f["up_mean_lag0"][0, 1] == 10.0
    assert np.isnan(f["up_mean_lag0"][0, 0])
