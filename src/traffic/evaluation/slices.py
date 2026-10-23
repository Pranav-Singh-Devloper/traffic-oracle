"""Evaluation slices: horizon, peak/off-peak, weekend, congested/free-flow, per sensor.

Peak status is judged on the TARGET timestamp (LA local time as stored).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from traffic.evaluation.io import Predictions
from traffic.evaluation.metrics import METRICS


def _minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:])


def slice_masks(p: Predictions, cfg: dict) -> dict[str, np.ndarray]:
    """Boolean masks (shape [S, F, N]) for each named slice; combine with p.mask later."""
    ts = pd.DatetimeIndex(p.target_ts.ravel())
    hm = (ts.hour * 60 + ts.minute).to_numpy().reshape(p.target_ts.shape)
    wk = (ts.day_of_week.to_numpy() < 5).reshape(p.target_ts.shape)
    am_lo, am_hi = (_minutes(x) for x in cfg["peak_windows"]["am"])
    pm_lo, pm_hi = (_minutes(x) for x in cfg["peak_windows"]["pm"])
    am = wk & (hm >= am_lo) & (hm < am_hi)
    pm = wk & (hm >= pm_lo) & (hm < pm_hi)
    peak = am | pm
    n = p.y_true.shape[-1]

    def b(x):  # [S, F] -> [S, F, N] view
        return np.broadcast_to(x[..., None], x.shape + (n,))

    thr = cfg["congestion_threshold_mph"]
    return {
        "all": np.ones_like(p.mask, dtype=bool),
        "peak_am": b(am),
        "peak_pm": b(pm),
        "peak": b(peak),
        "offpeak_weekday": b(wk & ~peak),
        "weekend": b(~wk),
        "congested": p.y_true < thr,
        "freeflow": p.y_true >= thr,
    }


def evaluate(p: Predictions, cfg: dict) -> pd.DataFrame:
    """Long table: model, seed, slice, horizon, metric, value, n_obs (NaN horizons skipped)."""
    masks = slice_masks(p, cfg)
    rows = []
    f = p.y_pred.shape[1]
    for h in range(1, f + 1):
        yh = p.y_pred[:, h - 1]
        if np.isnan(yh).all():
            continue
        yt = p.y_true[:, h - 1]
        for sname, sm in masks.items():
            m = p.mask[:, h - 1] & sm[:, h - 1]
            n_obs = int((m & ~np.isnan(yh)).sum())
            for mname, fn in METRICS.items():
                v = fn(yt, yh, m) if n_obs else np.nan
                rows.append((p.name, p.seed, sname, h, mname, v, n_obs))
    return pd.DataFrame(
        rows, columns=["model", "seed", "slice", "horizon", "metric", "value", "n_obs"]
    )


def per_sensor_mae(p: Predictions, horizon: int) -> np.ndarray:
    """[N] masked MAE per sensor at one horizon (1-indexed)."""
    yh, yt, m = p.y_pred[:, horizon - 1], p.y_true[:, horizon - 1], p.mask[:, horizon - 1]
    m = m & ~np.isnan(yh)
    err = np.where(m, np.abs(yt - yh), 0.0).sum(0)
    return err / np.maximum(m.sum(0), 1)
