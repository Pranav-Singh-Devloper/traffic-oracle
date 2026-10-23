"""Diebold-Mariano test (HAC variance) and block bootstrap CIs.

Test windows have stride 1, so per-timestamp losses are strongly autocorrelated:
a plain t-test would be over-confident. We use a Newey-West (Bartlett) long-run variance.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _loss_series(err: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean absolute error over sensors per anchor; returns (series, valid_anchor)."""
    cnt = mask.sum(1)
    s = np.where(mask, np.abs(err), 0.0).sum(1) / np.maximum(cnt, 1)
    return s, cnt > 0


def diebold_mariano(
    y: np.ndarray,
    yhat1: np.ndarray,
    yhat2: np.ndarray,
    mask: np.ndarray,
    h: int,
    max_lag: int | None = None,
) -> tuple[float, float]:
    """DM test on one horizon. Inputs are [S, N]. Returns (statistic, two-sided p-value).

    d_t = loss1_t - loss2_t with absolute-error loss averaged over sensors observed in
    both models. Negative statistic: model 1 has the lower loss. Long-run variance:
    Bartlett kernel with L = max(h-1, ceil(T^(1/3))) lags; Harvey-Leybourne-Newbold
    small-sample correction and a t(T-1) reference distribution.
    """
    m = mask & ~np.isnan(yhat1) & ~np.isnan(yhat2)
    l1, v1 = _loss_series(y - yhat1, m)
    l2, v2 = _loss_series(y - yhat2, m)
    ok = v1 & v2
    d = (l1 - l2)[ok]
    t = len(d)
    if t < 10 or np.allclose(d, 0):
        return (0.0, 1.0)
    lag = max_lag if max_lag is not None else max(h - 1, int(np.ceil(t ** (1 / 3))))
    dc = d - d.mean()
    lrv = (dc @ dc) / t
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / t
    lrv = max(lrv, 1e-12)
    dm = d.mean() / np.sqrt(lrv / t)
    corr = np.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    dm *= corr
    p = 2 * (1 - stats.t.cdf(abs(dm), df=t - 1))
    return float(dm), float(p)


def block_bootstrap_ci(
    abs_err_sum: np.ndarray,
    count: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """CI for masked MAE, resampling whole blocks (days).

    ``abs_err_sum``/``count`` are per-block sums for one horizon, shape [B].
    """
    rng = np.random.default_rng(seed)
    b = len(abs_err_sum)
    idx = rng.integers(0, b, size=(n_boot, b))
    mae = abs_err_sum[idx].sum(1) / np.maximum(count[idx].sum(1), 1)
    return float(np.quantile(mae, alpha / 2)), float(np.quantile(mae, 1 - alpha / 2))


def day_blocks(
    target_ts: np.ndarray, y: np.ndarray, yhat: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-day |error| sums and counts for one horizon ([S, N] inputs, [S] timestamps)."""
    import pandas as pd

    day = pd.DatetimeIndex(target_ts).normalize()
    codes, _ = pd.factorize(day)
    m = mask & ~np.isnan(yhat)
    err = np.where(m, np.abs(y - yhat), 0.0).sum(1)
    cnt = m.sum(1)
    nb = codes.max() + 1
    return np.bincount(codes, err, nb), np.bincount(codes, cnt, nb).astype(float)
