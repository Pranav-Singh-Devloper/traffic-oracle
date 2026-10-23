"""Shared fixtures: synthetic METR-LA-like data so CI never needs the real dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAW = Path("data/raw/metr-la.h5")


def pytest_collection_modifyitems(config, items):
    if RAW.exists():
        return
    skip = pytest.mark.skip(reason="data/raw/metr-la.h5 not present")
    for item in items:
        if "real_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def synthetic_metr() -> pd.DataFrame:
    """3 days x 10 sensors, daily sinusoid 20-65 mph, ~5% zeros, one 2h whole-network outage."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2012-03-01", periods=3 * 288, freq="5min")
    t = np.arange(len(idx))
    base = 42.5 + 22.5 * np.sin(2 * np.pi * t / 288)[:, None]
    x = base + rng.normal(0, 2, (len(idx), 10))
    x = np.clip(x, 20, 65)
    x[rng.random(x.shape) < 0.05] = 0.0
    x[100:124] = 0.0
    return pd.DataFrame(x, index=idx, columns=[str(i) for i in range(10)])


@pytest.fixture
def synthetic_graph():
    """Asymmetric weighted 10x10 adjacency plus sensor ids."""
    rng = np.random.default_rng(1)
    w = rng.random((10, 10)) * (rng.random((10, 10)) < 0.25)
    np.fill_diagonal(w, 1.0)
    return [str(i) for i in range(10)], w
