"""Masked StandardScaler fit on training timestamps only (observed values only)."""

from __future__ import annotations

import numpy as np


class MaskedStandardScaler:
    """z-score using only observed (mask == True) training values.

    Why: zeros are outages; including them would drag the mean down and inflate std.
    """

    def fit(self, x: np.ndarray, mask: np.ndarray, per_sensor: bool = False):
        m = mask.astype(float)
        if per_sensor:
            cnt = np.maximum(m.sum(0), 1)
            mean = (x * m).sum(0) / cnt
            std = np.sqrt((((x - mean) ** 2) * m).sum(0) / cnt)
            std = np.where(std < 1e-8, 1.0, std)
        else:
            mean = float(x[mask].mean())
            std = float(x[mask].std())
            std = std if std > 1e-8 else 1.0
        self.mean, self.std, self.per_sensor = mean, std, per_sensor
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def params(self) -> dict:
        return {"mean": np.asarray(self.mean).tolist(), "std": np.asarray(self.std).tolist()}
