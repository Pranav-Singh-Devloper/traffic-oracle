"""End-to-end data preparation shared by ML and DL: load -> mask -> split -> scale -> windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from traffic.data.load import MetrLA, load_all
from traffic.data.missing import build_mask, impute_inputs, tod_means
from traffic.data.scaling import MaskedStandardScaler
from traffic.data.split import SplitIndex, chronological_split, dcrnn_sample_split
from traffic.data.windows import Windows, build_channels, make_windows


@dataclass
class Prepared:
    data: MetrLA
    speed: np.ndarray  # [T, N] raw mph
    mask: np.ndarray  # [T, N]
    index: pd.DatetimeIndex
    split: SplitIndex  # time-axis split
    scaler: MaskedStandardScaler
    tod_mean: np.ndarray
    chan: np.ndarray
    windows: dict[str, Windows]


def prepare(cfg: dict, raw_dir: str = "data/raw", debug: bool = False) -> Prepared:
    """Build leak-free windows. Debug mode keeps a few sensors and days."""
    d = load_all(raw_dir)
    speed_df = d.speed
    if debug:
        n = cfg["debug"]["n_sensors"]
        speed_df = speed_df.iloc[: cfg["debug"]["n_days"] * 288 * 5, :n]
        d = MetrLA(speed_df, d.sensor_ids[:n], d.adj[:n, :n], d.distances, d.locations.iloc[:n])
    speed = speed_df.to_numpy(dtype=np.float64)
    index = speed_df.index
    mask = build_mask(speed)
    split = chronological_split(
        len(index), tuple(cfg["split"][k] for k in ("train", "val", "test"))
    )

    # Everything below is fit on TRAIN timestamps only.
    scaler = MaskedStandardScaler().fit(
        speed[split.train], mask[split.train], per_sensor=cfg["scaler"] == "per_sensor"
    )
    tm = tod_means(speed, mask, index, split.train)
    x_in = impute_inputs(speed, mask, index, tm, cfg["max_ffill_steps"], cfg["input_imputation"])
    x_scaled = scaler.transform(x_in)
    if cfg["input_imputation"] == "zero_mask":
        x_scaled = np.where(mask, x_scaled, 0.0)  # missing -> train mean in scaled space
    chan = build_channels(x_scaled, mask, index)

    H, F = cfg["history_steps"], cfg["horizon_steps"]
    if cfg["split_protocol"] == "strict":
        windows = {
            k: make_windows(chan, speed, mask, index, H, F, s.start, s.stop)
            for k, s in split.as_dict().items()
        }
    else:  # DCRNN: all windows first, then split the samples
        allw = make_windows(chan, speed, mask, index, H, F)
        ss = dcrnn_sample_split(len(allw))
        windows = {
            k: Windows(chan, speed, mask, index, allw.anchors[s], H, F)
            for k, s in ss.as_dict().items()
        }
    return Prepared(d, speed, mask, index, split, scaler, tm, chan, windows)
