import numpy as np
import pandas as pd

from traffic.data.windows import CHANNELS, build_channels, make_windows
from traffic.features.temporal import time_features


def _toy(t=60, n=3):
    idx = pd.date_range("2012-03-01", periods=t, freq="5min")
    y = np.arange(t * n, dtype=float).reshape(t, n) + 1
    return idx, y


def test_alignment_exact():
    idx, y = _toy()
    chan = build_channels(y, y > 0, idx)
    w = make_windows(chan, y, y > 0, idx, H=12, F=12)
    ty, _ = w.target(0)
    np.testing.assert_array_equal(ty, y[12:24])
    np.testing.assert_array_equal(w.x(0)[..., 0], y[0:12])
    assert len(w) == 60 - 12 - 12 + 1
    X, Y, M = w.materialize()
    np.testing.assert_array_equal(X[3], w.x(3))
    np.testing.assert_array_equal(Y[3], w.target(3)[0])


def test_split_bounds_respected():
    idx, y = _toy()
    chan = build_channels(y, y > 0, idx)
    w = make_windows(chan, y, y > 0, idx, 12, 12, start=20, stop=50)
    assert w.input_ts().min() >= idx[20] and w.target_ts().max() <= idx[49]


def test_channel_order_and_holiday():
    idx, y = _toy()
    chan = build_channels(y, y > 0, idx)
    assert chan.shape[-1] == len(CHANNELS)
    assert CHANNELS[:2] == ["speed", "is_missing"]
    d = pd.date_range("2012-05-27", "2012-05-29", freq="1D")
    tf = time_features(d)
    assert tf["is_holiday"].tolist() == [0.0, 1.0, 0.0]  # Memorial Day 2012-05-28
    # 23:55 and 00:00 are close under cyclic encoding
    e = time_features(pd.DatetimeIndex(["2012-03-01 23:55", "2012-03-02 00:00"]))
    assert abs(e["tod_sin"].diff().iloc[1]) < 0.03
