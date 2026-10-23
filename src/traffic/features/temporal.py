"""Cyclic time features, weekend, holiday.

Cyclic (sin/cos) encoding keeps 23:55 and 00:00 close in feature space.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

TIME_COLS = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday"]


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    minute = index.hour * 60 + index.minute
    dow = index.day_of_week
    us = holidays.US(years=sorted(set(index.year)))
    return pd.DataFrame(
        {
            "tod_sin": np.sin(2 * np.pi * minute / 1440),
            "tod_cos": np.cos(2 * np.pi * minute / 1440),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "is_weekend": (dow >= 5).astype(float),
            "is_holiday": np.array([d in us for d in index.date], dtype=float),
        },
        index=index,
    )
