"""Sliding windows (history H, horizon F) built INSIDE each split.

Channels C: scaled speed, is_missing, then the time features.
Windows keep references to the [T, N, C] array and only store anchors, so the
~2 GB of overlapping copies is never materialised unless asked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from traffic.features.temporal import TIME_COLS, time_features

CHANNELS = ["speed", "is_missing", *TIME_COLS]


def build_channels(
    speed_scaled: np.ndarray, mask: np.ndarray, index: pd.DatetimeIndex
) -> np.ndarray:
    """[T, N, C] float32 input tensor in the order of ``CHANNELS``."""
    t, n = speed_scaled.shape
    tf = time_features(index).to_numpy(dtype=np.float32)  # [T, 6]
    out = np.empty((t, n, len(CHANNELS)), dtype=np.float32)
    out[..., 0] = speed_scaled
    out[..., 1] = ~mask
    out[..., 2:] = tf[:, None, :]
    return out


@dataclass
class Windows:
    chan: np.ndarray  # [T, N, C]
    y: np.ndarray  # [T, N] original mph
    ymask: np.ndarray  # [T, N] bool
    index: pd.DatetimeIndex
    anchors: np.ndarray  # last input step of each window
    H: int
    F: int

    def __len__(self) -> int:
        return len(self.anchors)

    def x(self, i: int) -> np.ndarray:
        t = self.anchors[i]
        return self.chan[t - self.H + 1 : t + 1]  # [H, N, C]

    def target(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        t = self.anchors[i]
        return self.y[t + 1 : t + self.F + 1], self.ymask[t + 1 : t + self.F + 1]

    def input_ts(self) -> np.ndarray:
        """[S, H] timestamps of inputs."""
        off = np.arange(-self.H + 1, 1)
        return self.index.values[self.anchors[:, None] + off]

    def target_ts(self) -> np.ndarray:
        """[S, F] timestamps of targets."""
        off = np.arange(1, self.F + 1)
        return self.index.values[self.anchors[:, None] + off]

    def materialize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """X [S,H,N,C], Y [S,F,N], M [S,F,N] (uses a lot of memory on the full set)."""
        hoff = np.arange(-self.H + 1, 1)
        foff = np.arange(1, self.F + 1)
        a = self.anchors
        return (
            self.chan[a[:, None] + hoff],
            self.y[a[:, None] + foff],
            self.ymask[a[:, None] + foff],
        )


def make_windows(
    chan: np.ndarray,
    y: np.ndarray,
    ymask: np.ndarray,
    index: pd.DatetimeIndex,
    H: int = 12,
    F: int = 12,
    start: int = 0,
    stop: int | None = None,
) -> Windows:
    """Anchors t with inputs t-H+1..t and targets t+1..t+F all inside [start, stop)."""
    stop = len(index) if stop is None else stop
    anchors = np.arange(start + H - 1, stop - F)
    return Windows(chan, y, ymask, index, anchors, H, F)
