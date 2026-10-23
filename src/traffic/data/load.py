"""Read metr-la.h5, adj_mx.pkl and CSVs; integrity asserts.

All arrays in the project use the sensor order of ``adj_mx.pkl``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_SENSORS = 207
N_STEPS = 34272  # 119 days * 288


@dataclass
class MetrLA:
    speed: pd.DataFrame  # [T, N] mph, 0 == missing, columns in pkl order
    sensor_ids: list[str]
    adj: np.ndarray  # canonical adj_mx.pkl, [N, N]
    distances: pd.DataFrame  # columns from, to, cost (filtered to our sensors)
    locations: pd.DataFrame  # sensor_id (str), latitude, longitude, in pkl order


def _read_hdf_fallback(path: Path) -> pd.DataFrame:
    """Read the pandas fixed-format file with PyTables.

    Why: ``pd.read_hdf`` fails on this 2019 file with recent pandas/PyTables
    (bytes-vs-str attribute error), but the data layout is plain.
    """
    import tables

    with tables.open_file(str(path)) as f:
        g = f.root.df
        cols = [c.decode() if isinstance(c, bytes) else str(c) for c in g.block0_items[:]]
        axis0 = [c.decode() if isinstance(c, bytes) else str(c) for c in g.axis0[:]]
        index = pd.to_datetime(g.axis1[:], unit="ns")
        values = g.block0_values[:]
    assert cols == axis0
    return pd.DataFrame(values, index=index, columns=cols)


def load_speed(path: str | Path) -> pd.DataFrame:
    """Load the speed table (key ``df``); columns are sensor ids as ``str``."""
    path = Path(path)
    try:
        df = pd.read_hdf(path)
    except Exception:  # noqa: BLE001
        df = _read_hdf_fallback(path)
    df.columns = [str(c) for c in df.columns]
    return df


def load_adj_pkl(path: str | Path) -> tuple[list[str], dict[str, int], np.ndarray]:
    """Load ``(sensor_ids, sensor_id_to_ind, adj_mx)``; ids normalised to ``str``."""
    with open(path, "rb") as f:
        ids, ind, adj = pickle.load(f, encoding="latin1")
    return [str(i) for i in ids], {str(k): v for k, v in ind.items()}, np.asarray(adj)


def load_distances(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["from"], d["to"] = d["from"].astype(str), d["to"].astype(str)
    return d


def load_locations(path: str | Path) -> pd.DataFrame:
    loc = pd.read_csv(path)
    loc["sensor_id"] = loc["sensor_id"].astype(str)
    return loc


def check_speed(df: pd.DataFrame) -> None:
    """Integrity asserts from SPEC section 2."""
    assert df.shape == (N_STEPS, N_SENSORS), f"unexpected shape {df.shape}"
    assert N_STEPS == 119 * 288
    assert df.index[0] == pd.Timestamp("2012-03-01 00:00:00"), df.index[0]
    assert df.index[-1] == pd.Timestamp("2012-06-27 23:55:00"), df.index[-1]
    steps = np.diff(df.index.values).astype("timedelta64[m]").astype(int)
    assert (steps == 5).all(), "index is not a regular 5-minute grid"
    assert not df.isna().to_numpy().any(), "unexpected NaN (zeros are the missing marker)"
    assert (df.to_numpy() >= 0).all(), "negative speeds"


def load_all(raw_dir: str | Path = "data/raw") -> MetrLA:
    """Load everything, assert integrity, align columns to the pkl sensor order."""
    raw = Path(raw_dir)
    speed = load_speed(raw / "metr-la.h5")
    check_speed(speed)
    ids, _, adj = load_adj_pkl(raw / "adj_mx.pkl")
    assert adj.shape == (N_SENSORS, N_SENSORS) and len(ids) == N_SENSORS
    txt = [s for s in (raw / "graph_sensor_ids.txt").read_text().strip().split(",") if s]
    assert txt == ids, "graph_sensor_ids.txt disagrees with adj_mx.pkl"
    if list(speed.columns) != ids:
        assert set(speed.columns) == set(ids), "sensor sets differ"
        print("NOTE: reordering speed columns to adj_mx.pkl order")
        speed = speed[ids]
    dist = load_distances(raw / "distances_la_2012.csv")
    dist = dist[dist["from"].isin(ids) & dist["to"].isin(ids)].reset_index(drop=True)
    loc = load_locations(raw / "graph_sensor_locations.csv").set_index("sensor_id").loc[ids]
    assert loc[["latitude", "longitude"]].notna().all().all()
    return MetrLA(speed, ids, adj, dist, loc.reset_index())
