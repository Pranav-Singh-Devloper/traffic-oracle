"""Naive baselines: persistence, historical average, seasonal naive.

A learned model that cannot beat persistence at 15 minutes is not useful.
All functions return predictions of shape [S, F, N] in mph for the given anchors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from traffic.data.missing import impute_inputs, slot_of_day, tod_means


def persistence(
    speed: np.ndarray,
    mask: np.ndarray,
    index: pd.DatetimeIndex,
    train_idx: slice,
    anchors: np.ndarray,
    history: int,
    horizon: int,
) -> np.ndarray:
    """y_hat(t+h) = last OBSERVED value in the input window.

    Uses the past only: forward-fill up to ``history-1`` steps, then the train-only
    time-of-day mean if the whole window is missing.
    """
    tm = tod_means(speed, mask, index, train_idx)
    last = impute_inputs(speed, mask, index, tm, max_ffill=history - 1)
    return np.repeat(last[anchors][:, None, :], horizon, axis=1)


def historical_table(
    speed: np.ndarray, mask: np.ndarray, index: pd.DatetimeIndex, train_idx: slice
) -> np.ndarray:
    """[7, 288, N] mean observed speed per (day-of-week, slot) on TRAIN only.

    Cells without train observations fall back to the (slot) train mean.
    """
    sp, mk = speed[train_idx], mask[train_idx]
    ix = index[train_idx]
    dow, slot = ix.day_of_week.to_numpy(), slot_of_day(ix)
    fallback = tod_means(speed, mask, index, train_idx)
    out = np.empty((7, 288, speed.shape[1]))
    for d in range(7):
        for s in range(288):
            sel = (dow == d) & (slot == s)
            cnt = mk[sel].sum(0)
            tot = np.where(mk[sel], sp[sel], 0.0).sum(0)
            out[d, s] = np.where(cnt > 0, tot / np.maximum(cnt, 1), fallback[s])
    return out


def historical_average(table: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """Look up the (day-of-week, slot) mean for each target timestamp ([S, F] -> [S, F, N])."""
    ts = pd.DatetimeIndex(target_ts.ravel())
    dow = ts.day_of_week.to_numpy()
    slot = slot_of_day(ts)
    return table[dow, slot].reshape(target_ts.shape + (table.shape[-1],))


def seasonal_naive(
    speed: np.ndarray,
    mask: np.ndarray,
    anchors: np.ndarray,
    horizon: int,
    hist_pred: np.ndarray,
    period: int = 288,
) -> tuple[np.ndarray, float]:
    """y_hat(t+h) = y(t+h-288), falling back to the historical average when missing.

    Looks outside the 12-step window (allowed for a baseline; it only uses the past).
    Returns (predictions, share of entries that needed the fallback).
    """
    offs = np.arange(1, horizon + 1)
    src = anchors[:, None] + offs[None, :] - period  # [S, F]
    if src.min() < 0:
        raise ValueError("seasonal naive needs at least one day of history before the anchors")
    y, m = speed[src], mask[src]  # [S, F, N]
    out = np.where(m, y, hist_pred)
    return out, float((~m).mean())
