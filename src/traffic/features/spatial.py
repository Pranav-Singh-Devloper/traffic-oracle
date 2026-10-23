"""Neighbor features from the road graph (for XGB-spatial).

Convention (ADR-002): ``W[i, j] > 0`` means node i aggregates from node j; in ``W`` those are
DOWNSTREAM sensors, in ``W.T`` UPSTREAM sensors. Nodes with no neighbor in a direction get NaN.
"""

from __future__ import annotations

import numpy as np


def _row_norm(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = a.copy()
    np.fill_diagonal(a, 0.0)
    s = a.sum(1)
    has = s > 0
    out = np.zeros_like(a)
    out[has] = a[has] / s[has, None]
    return out, has


def neighbor_operators(w: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Row-normalised operators (matrix, has_neighbor) for down, up and 2-hop neighbors."""
    down, h_d = _row_norm(w)
    up, h_u = _row_norm(w.T)
    b = (((w + w.T) > 0) & ~np.eye(len(w), dtype=bool)).astype(float)
    two = ((b @ b) > 0) & (b == 0) & ~np.eye(len(w), dtype=bool)
    hop2, h_2 = _row_norm(two.astype(float))
    return {"down": (down, h_d), "up": (up, h_u), "hop2": (hop2, h_2)}


def neighbor_features(
    x_in: np.ndarray, w: np.ndarray, anchors: np.ndarray, lags: tuple[int, ...] = (0, 1, 2)
) -> dict[str, np.ndarray]:
    """Adjacency-weighted neighbor speeds at the anchor (and lags), 2-hop mean, differences."""
    ops = neighbor_operators(w)
    out: dict[str, np.ndarray] = {}

    def agg(name: str, k: int) -> np.ndarray:
        mat, has = ops[name]
        v = x_in[anchors - k] @ mat.T  # [A, N]
        v = v.astype(np.float32)
        v[:, ~has] = np.nan
        return v

    for k in lags:
        out[f"up_mean_lag{k}"] = agg("up", k)
        out[f"down_mean_lag{k}"] = agg("down", k)
    out["hop2_mean_lag0"] = agg("hop2", 0)
    own = x_in[anchors].astype(np.float32)
    out["own_minus_up"] = own - out["up_mean_lag0"]
    out["own_minus_down"] = own - out["down_mean_lag0"]
    return out
