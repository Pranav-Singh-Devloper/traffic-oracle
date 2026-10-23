import numpy as np
import pandas as pd

from traffic.evaluation.io import Predictions, assert_aligned
from traffic.evaluation.slices import evaluate, slice_masks
from traffic.models.baselines import historical_average, historical_table, persistence


def test_persistence_uses_last_observed(synthetic_metr):
    x = synthetic_metr.to_numpy()
    m = x > 0
    idx = synthetic_metr.index
    anchors = np.array([300, 400])
    out = persistence(x, m, idx, slice(0, 600), anchors, 12, 12)
    assert out.shape == (2, 12, x.shape[1])
    j = 0
    last = [x[t - k, j] for t in [300] for k in range(12) if x[t - k, j] > 0]
    if m[300, j]:
        assert out[0, 0, j] == x[300, j]
    elif last:
        assert out[0, 0, j] == last[0]


def test_historical_average_train_only(synthetic_metr):
    x = synthetic_metr.to_numpy()
    m = x > 0
    idx = synthetic_metr.index
    t1 = historical_table(x, m, idx, slice(0, 500))
    x2 = x.copy()
    x2[500:] += 50
    t2 = historical_table(x2, m, idx, slice(0, 500))
    np.testing.assert_allclose(t1, t2, atol=1e-9)
    ts = idx[600:606].values.reshape(1, -1)
    assert historical_average(t1, ts).shape == (1, 6, x.shape[1])


def _preds(synthetic_metr):
    idx = synthetic_metr.index
    s, f, n = 40, 12, 4
    ts = np.array([idx[i : i + f].values for i in range(s)])
    y = np.random.default_rng(0).uniform(20, 65, (s, f, n)).astype(np.float32)
    return Predictions("t", None, y + 1, y, np.ones_like(y, bool), ts, np.array(list("abcd")), {})


def test_slices_and_evaluate(synthetic_metr):
    p = _preds(synthetic_metr)
    cfg = {
        "peak_windows": {"am": ["06:00", "10:00"], "pm": ["15:00", "19:00"]},
        "congestion_threshold_mph": 40,
    }
    masks = slice_masks(p, cfg)
    assert set(masks) >= {"all", "peak", "weekend", "congested", "freeflow"}
    assert (masks["congested"] ^ masks["freeflow"]).all()
    df = evaluate(p, cfg)
    row = df[(df.slice == "all") & (df.horizon == 3) & (df.metric == "MAE")]
    assert row.value.iloc[0] == pytest_approx(1.0)


def pytest_approx(x):
    import pytest

    return pytest.approx(x, abs=1e-4)


def test_assert_aligned_detects_mismatch(synthetic_metr):
    a = _preds(synthetic_metr)
    b = _preds(synthetic_metr)
    assert_aligned(a, b)
    b.mask = b.mask.copy()
    b.mask[0, 0, 0] = False
    import pytest

    with pytest.raises(ValueError):
        assert_aligned(a, b)
    assert isinstance(pd.Timestamp("2012-03-01"), pd.Timestamp)
