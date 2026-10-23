"""Run the three baselines on the test split and save predictions.

Usage: python scripts/run_baselines.py [--debug]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from traffic.config import load_config
from traffic.data.pipeline import prepare
from traffic.evaluation.io import save_predictions
from traffic.logging_utils import log_run
from traffic.models import baselines as bl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = prepare(cfg, cfg["paths"]["raw"], debug=args.debug)
    w = p.windows["test"]
    y, m = w.targets()
    ts = w.target_ts()
    h, f = cfg["history_steps"], cfg["horizon_steps"]
    out = Path(cfg["paths"]["results"]) / ("debug" if args.debug else "") / "predictions"
    ids = p.data.sensor_ids
    meta = {"protocol": cfg["split_protocol"], "H": h, "split": "test"}

    pers = bl.persistence(p.speed, p.mask, p.index, p.split.train, w.anchors, h, f)
    table = bl.historical_table(p.speed, p.mask, p.index, p.split.train)
    hist = bl.historical_average(table, ts)
    seas, fallback = bl.seasonal_naive(p.speed, p.mask, w.anchors, f, hist)
    meta_s = {**meta, "seasonal_fallback_share": fallback}
    for name, pred, mt in (
        ("persistence", pers, meta),
        ("historical_average", hist, meta),
        ("seasonal_naive", seas, meta_s),
    ):
        save_predictions(out / f"{name}.npz", pred, y, m, ts, ids, {**mt, "model": name})
        print(f"saved {name}")
    print(f"seasonal naive fallback share: {fallback:.4f}")
    log_run("baselines", cfg, {"seasonal_fallback_share": fallback})


if __name__ == "__main__":
    main()
