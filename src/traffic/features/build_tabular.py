"""Assemble tabular design matrices for the ML models.

X has shape [A, N, F] (anchor, sensor, feature); flatten to [A*N, F] for pooled models and
slice ``X[:, j]`` for per-segment models. Targets are y[t+h, i] with missing ones masked out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from traffic.data.pipeline import Prepared
from traffic.features.lags import lag_features
from traffic.features.spatial import neighbor_features
from traffic.features.temporal import time_features


@dataclass
class Tabular:
    X: np.ndarray  # [A, N, F] float32
    names: list[str]
    anchors: np.ndarray  # [A]
    static_cols: list[int]  # feature indices that are static per sensor
    categorical: list[int]  # feature indices that are categorical (sensor_idx)

    def flat(self) -> np.ndarray:
        return self.X.reshape(-1, self.X.shape[-1])


def static_features(p: Prepared) -> dict[str, np.ndarray]:
    w = p.data.adj.copy()
    np.fill_diagonal(w, 0)
    loc = p.data.locations
    return {
        "sensor_idx": np.arange(w.shape[0], dtype=np.float32),
        "lat": loc["latitude"].to_numpy(np.float32),
        "lon": loc["longitude"].to_numpy(np.float32),
        "in_degree": (w > 0).sum(0).astype(np.float32),
        "out_degree": (w > 0).sum(1).astype(np.float32),
    }


def build_tabular(
    p: Prepared,
    split: str,
    cfg: dict,
    kind: str = "global",
    spatial: bool = False,
) -> Tabular:
    """Features for all anchors of ``split``.

    ``kind='segment'`` drops static features (constant within one sensor's model).
    """
    anchors = p.windows[split].anchors
    h = cfg["history_steps"]
    feats = lag_features(p.x_in, p.mask, anchors, h, cfg.get("use_daily_lag", False))
    n = p.x_in.shape[1]
    tf = time_features(p.index[anchors])
    for c in tf.columns:
        feats[c] = np.repeat(tf[c].to_numpy(np.float32)[:, None], n, axis=1)
    static_names: list[str] = []
    if kind == "global":
        for k, v in static_features(p).items():
            feats[k] = np.broadcast_to(v[None, :], (len(anchors), n)).astype(np.float32)
            static_names.append(k)
    if spatial:
        feats.update(neighbor_features(p.x_in, p.data.adj, anchors))
    names = list(feats)
    x = np.stack([feats[k] for k in names], axis=-1).astype(np.float32)
    cat = [names.index("sensor_idx")] if "sensor_idx" in names else []
    return Tabular(x, names, anchors, [names.index(k) for k in static_names], cat)


def targets(p: Prepared, split: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """y [A, N] mph and observed mask [A, N] at the given horizon (1-indexed)."""
    a = p.windows[split].anchors + horizon
    return p.speed[a], p.mask[a]


def subsample_rows(mask_flat: np.ndarray, n_rows: int | None, frac: float, seed: int) -> np.ndarray:
    """Indices of usable (observed-target) rows, optionally randomly subsampled."""
    idx = np.flatnonzero(mask_flat)
    rng = np.random.default_rng(seed)
    if frac < 1.0:
        idx = rng.choice(idx, size=int(len(idx) * frac), replace=False)
    if n_rows is not None and len(idx) > n_rows:
        idx = rng.choice(idx, size=n_rows, replace=False)
    return np.sort(idx)


def scatter_to_windows(
    pred_flat: np.ndarray,
    anchors: np.ndarray,
    n_sensors: int,
    p_index: pd.DatetimeIndex | None = None,
) -> np.ndarray:
    """[A*N] -> [A, N] (inverse of ``flat``)."""
    return pred_flat.reshape(len(anchors), n_sensors)
