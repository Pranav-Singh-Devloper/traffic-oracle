"""Common prediction format shared by every model.

``results/predictions/{model}[_seed{s}].npz`` holds y_pred/y_true [S, F, N] in mph,
mask [S, F, N], target_ts [S, F] (ns), sensor_ids [N] and a JSON ``meta`` string.
Horizons a model does not produce are stored as NaN and skipped by the evaluator.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Predictions:
    name: str
    seed: int | None
    y_pred: np.ndarray
    y_true: np.ndarray
    mask: np.ndarray
    target_ts: np.ndarray
    sensor_ids: np.ndarray
    meta: dict


def save_predictions(
    path: str | Path,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray,
    target_ts: np.ndarray,
    sensor_ids: list[str],
    meta: dict | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_pred=y_pred.astype(np.float32),
        y_true=y_true.astype(np.float32),
        mask=mask.astype(bool),
        target_ts=target_ts.astype("datetime64[ns]").astype(np.int64),
        sensor_ids=np.asarray(sensor_ids, dtype=str),
        meta=json.dumps(meta or {}),
    )


def load_predictions(path: str | Path) -> Predictions:
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    m = re.match(r"(.+)_seed(\d+)$", path.stem)
    name, seed = (m.group(1), int(m.group(2))) if m else (path.stem, None)
    return Predictions(
        name,
        seed,
        z["y_pred"],
        z["y_true"],
        z["mask"],
        z["target_ts"].astype("datetime64[ns]"),
        z["sensor_ids"],
        json.loads(str(z["meta"])),
    )


def assert_aligned(a: Predictions, b: Predictions) -> None:
    """Fairness rule: same test timestamps, same masks, same sensors for every model."""
    if a.y_pred.shape != b.y_pred.shape:
        raise ValueError(f"{a.name} vs {b.name}: shape {a.y_pred.shape} != {b.y_pred.shape}")
    if not np.array_equal(a.target_ts, b.target_ts):
        raise ValueError(f"{a.name} vs {b.name}: target timestamps differ")
    if not np.array_equal(a.mask, b.mask):
        raise ValueError(f"{a.name} vs {b.name}: masks differ")
    if not np.array_equal(a.sensor_ids, b.sensor_ids):
        raise ValueError(f"{a.name} vs {b.name}: sensor ids differ")
