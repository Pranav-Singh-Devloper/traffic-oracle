"""Mask construction, outage stats, input imputation.

Targets are NEVER imputed. Only model inputs pass through ``impute_inputs``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STEPS_PER_DAY = 288


def build_mask(speed: np.ndarray) -> np.ndarray:
    """True where the sensor reported; speed == 0 means outage, not stopped traffic."""
    return speed > 0


def slot_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Time-of-day slot 0..287 for each timestamp."""
    return ((index.hour * 60 + index.minute) // 5).to_numpy()


def outage_stats(mask: np.ndarray, index: pd.DatetimeIndex) -> dict:
    """Overall/per-sensor missing rates, longest outage runs, whole-network outages."""
    miss = ~mask
    per_sensor = miss.mean(0)
    longest = []
    for j in range(miss.shape[1]):
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, miss[:, j].astype(int), 0])))[::2]
        longest.append(int(runs.max()) if len(runs) else 0)
    net = miss.mean(1)
    return {
        "overall_missing_rate": float(miss.mean()),
        "per_sensor_missing_rate": per_sensor.tolist(),
        "longest_outage_steps_per_sensor": longest,
        "max_longest_outage_steps": int(max(longest)),
        "whole_network_outage_timestamps": int((net >= 0.9).sum()),
        "days_with_whole_network_outage": sorted({str(t.date()) for t in index[net >= 0.9]}),
    }


def tod_means(
    speed: np.ndarray, mask: np.ndarray, index: pd.DatetimeIndex, train_idx: slice
) -> np.ndarray:
    """[288, N] mean observed speed per time-of-day slot, TRAIN timestamps only.

    Slots/sensors without train observations fall back to the sensor's train mean,
    then the global train mean.
    """
    sp, mk = speed[train_idx], mask[train_idx]
    slots = slot_of_day(index[train_idx])
    n = speed.shape[1]
    out = np.full((STEPS_PER_DAY, n), np.nan)
    for s in range(STEPS_PER_DAY):
        sel = slots == s
        cnt = mk[sel].sum(0)
        tot = np.where(mk[sel], sp[sel], 0.0).sum(0)
        out[s] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    sensor_mean = np.where(mk, sp, 0.0).sum(0) / np.maximum(mk.sum(0), 1)
    global_mean = sp[mk].mean()
    sensor_mean = np.where(mk.sum(0) > 0, sensor_mean, global_mean)
    return np.where(np.isnan(out), sensor_mean[None, :], out)


def impute_inputs(
    speed: np.ndarray,
    mask: np.ndarray,
    index: pd.DatetimeIndex,
    tod_mean: np.ndarray,
    max_ffill: int = 6,
    method: str = "ffill_tod_mean",
) -> np.ndarray:
    """Input imputation (option a) or pass-through (option b, DCRNN convention).

    Forward-fill only looks at the past, so carrying a value across a split
    boundary does not leak the future.
    """
    if method == "zero_mask":
        return speed.copy()
    if method != "ffill_tod_mean":
        raise ValueError(method)
    df = pd.DataFrame(np.where(mask, speed, np.nan)).ffill(limit=max_ffill)
    arr = df.to_numpy()
    fill = tod_mean[slot_of_day(index)]
    return np.where(np.isnan(arr), fill, arr)
