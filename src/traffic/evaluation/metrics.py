"""Masked MAE / RMSE / MAPE, computed in original mph.

All-masked input returns NaN (never 0) so an empty slice cannot look like a perfect score.
NaN predictions (horizons a model does not produce) are removed from the mask.
"""

from __future__ import annotations

import warnings

import numpy as np

MAPE_MIN_SPEED = 1.0  # mph; MAPE blows up for tiny targets (ADR-005)


def _valid(y, yhat, m) -> np.ndarray:
    return np.asarray(m, bool) & ~np.isnan(yhat)


def masked_mae(y, yhat, m) -> float:
    """sum(m*|y-yhat|) / sum(m)."""
    v = _valid(y, yhat, m)
    if not v.any():
        warnings.warn("masked_mae: no valid entries", stacklevel=2)
        return float("nan")
    return float(np.abs(y[v] - yhat[v]).mean())


def masked_rmse(y, yhat, m) -> float:
    """sqrt(sum(m*(y-yhat)^2) / sum(m))."""
    v = _valid(y, yhat, m)
    if not v.any():
        warnings.warn("masked_rmse: no valid entries", stacklevel=2)
        return float("nan")
    return float(np.sqrt(((y[v] - yhat[v]) ** 2).mean()))


def masked_mape(y, yhat, m, min_y: float = MAPE_MIN_SPEED) -> float:
    """Percent error over entries with y >= min_y (default 1 mph)."""
    v = _valid(y, yhat, m) & (y >= min_y)
    if not v.any():
        warnings.warn("masked_mape: no valid entries", stacklevel=2)
        return float("nan")
    return float((np.abs(y[v] - yhat[v]) / y[v]).mean() * 100)


METRICS = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}
