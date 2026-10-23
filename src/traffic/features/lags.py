"""Lagged speeds, rolling stats and differences, computed at anchor times.

Every feature at anchor t uses data at times <= t only (tested in test_features_no_future).
"""

from __future__ import annotations

import numpy as np


def lag_features(
    x_in: np.ndarray,
    mask: np.ndarray,
    anchors: np.ndarray,
    history: int = 12,
    use_daily_lag: bool = False,
) -> dict[str, np.ndarray]:
    """Return {name: [A, N] float32}.

    ``x_in`` is the imputed input speed (mph) and ``mask`` the observed flags, both [T, N].
    The daily lag (t-288) lies outside the history window, so it is an explicit flag.
    """
    lags = np.stack([x_in[anchors - k] for k in range(history)], axis=0)  # [H, A, N]
    f: dict[str, np.ndarray] = {f"lag_{k}": lags[k] for k in range(history)}
    for w in (3, 6, 12):
        w = min(w, history)
        seg = lags[:w]
        f[f"roll_mean_{w}"] = seg.mean(0)
        f[f"roll_std_{w}"] = seg.std(0)
        f[f"roll_min_{w}"] = seg.min(0)
        f[f"roll_max_{w}"] = seg.max(0)
    f["diff_1"] = lags[0] - lags[1]
    f["diff_3"] = lags[0] - lags[3]
    miss = np.stack([~mask[anchors - k] for k in range(history)], axis=0).astype(np.float32)
    for k in range(history):
        f[f"miss_{k}"] = miss[k]
    f["n_missing_window"] = miss.sum(0)
    if use_daily_lag:
        src = anchors - 288
        v = x_in[np.clip(src, 0, None)].astype(np.float32)
        v[src < 0] = np.nan
        f["lag_288"] = v
    return {k: v.astype(np.float32) for k, v in f.items()}
