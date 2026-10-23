"""Chronological split by timestamp (strict protocol) and DCRNN sample split."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitIndex:
    train: slice
    val: slice
    test: slice

    def as_dict(self) -> dict[str, slice]:
        return {"train": self.train, "val": self.val, "test": self.test}


def chronological_split(
    n_steps: int, ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)
) -> SplitIndex:
    """First 70% train, next 10% val, last 20% test along the TIME axis."""
    n_train = int(n_steps * ratios[0])
    n_val = int(n_steps * ratios[1])
    return SplitIndex(
        slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_steps)
    )


def dcrnn_sample_split(n_samples: int) -> SplitIndex:
    """DCRNN rounding: test = round(0.2n), train = round(0.7n), val = rest (sample axis).

    Literature-comparable protocol only; windows near boundaries overlap across splits.
    """
    n_test = round(n_samples * 0.2)
    n_train = round(n_samples * 0.7)
    n_val = n_samples - n_train - n_test
    return SplitIndex(
        slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_samples)
    )


def boundaries(split: SplitIndex, index: pd.DatetimeIndex) -> dict[str, dict[str, str | int]]:
    """Human-readable first/last timestamp of each split."""
    out = {}
    for name, s in split.as_dict().items():
        out[name] = {
            "start": str(index[s.start]),
            "end": str(index[s.stop - 1]),
            "n_steps": s.stop - s.start,
        }
    return out
