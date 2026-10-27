"""Build app/demo_data/bundle.npz: a compact, git-trackable slice of the test set for cloud deployment.

Keeps every 6th test window (30-minute stride), float16 forecasts of the main models (seed-averaged),
the matching truth, the raw speeds around those windows, and the sensor locations.
Usage: python scripts/make_demo_bundle.py [--stride 6]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from traffic.config import load_config
from traffic.data.pipeline import prepare

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "predictions"
MODELS = [
    "persistence",
    "historical_average",
    "seasonal_naive",
    "xgb_per_segment",
    "xgb_global",
    "rf_multioutput",
    "xgb_spatial",
    "gru_nograph",
    "stgat",
    "hybrid_residual",
    "hybrid_gating",
    "hybrid_average",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=6)
    a = ap.parse_args()
    cfg = load_config(str(ROOT / "configs/base.yaml"))
    p = prepare(cfg)
    anchors = p.windows["test"].anchors
    sel = np.arange(0, len(anchors), a.stride)
    z = np.load(PRED / "persistence.npz")
    out = {
        "sel": sel,
        "y_true": z["y_true"][sel].astype(np.float16),
        "tts": z["target_ts"][sel],
        "stride": np.array(a.stride),
    }
    lo, hi = int(anchors[sel].min()) - 11, int(anchors[sel].max()) + 1
    out["slab_speed"] = p.speed[lo:hi].astype(np.float16)
    out["slab_start_ns"] = np.array(p.index[lo].value)
    out["anchor_pos"] = (anchors[sel] - lo).astype(np.int64)
    for m in MODELS:
        files = sorted(PRED.glob(f"{m}_seed*.npz")) or [PRED / f"{m}.npz"]
        arr = np.mean([np.load(f)["y_pred"][sel] for f in files if f.exists()], axis=0)
        out[f"pred_{m}"] = np.nan_to_num(arr, nan=0.0).astype(np.float16)
    loc = p.data.locations[["sensor_id", "latitude", "longitude"]]
    out["sensor_ids"] = np.array(loc.sensor_id.astype(str).tolist(), dtype="U12")
    out["lat"] = loc.latitude.to_numpy(np.float32)
    out["lon"] = loc.longitude.to_numpy(np.float32)
    dest = ROOT / "app" / "demo_data" / "bundle.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(
        f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB), {len(sel)} windows, {len(MODELS)} models"
    )


if __name__ == "__main__":
    main()
