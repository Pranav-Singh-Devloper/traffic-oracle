"""Congestion maps, error maps, space-time diagram, attention figures, key summary plots.

Needs predictions from xgb_global and stgat (seed 0 used for single-model maps) and the STGAT
checkpoint for attention. Usage: python scripts/make_maps.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from traffic.config import load_config
from traffic.data.pipeline import prepare
from traffic.evaluation.io import load_predictions
from traffic.evaluation.slices import slice_masks
from traffic.models.factory import build_model, channel_slice
from traffic.viz import attention_viz as av
from traffic.viz import congestion_map as cm
from traffic.viz import horizon_plots as hp

RES = Path("results")
MET, MAPS, FIG = RES / "metrics", RES / "maps", RES / "figures" / "maps"


def per_sensor_mae(p, h, sel=None) -> np.ndarray:
    m = p.mask[:, h - 1] & (True if sel is None else sel[:, h - 1])
    e = np.where(m, np.abs(p.y_true[:, h - 1] - p.y_pred[:, h - 1]), 0).sum(0)
    return e / np.maximum(m.sum(0), 1)


def main() -> None:
    cfg = load_config("configs/dl/stgat.yaml")
    p = prepare(cfg, cfg["paths"]["raw"])
    loc = p.data.locations
    pred = {n: load_predictions(RES / "predictions" / f"{n}.npz") for n in ("xgb_global",)}
    st = load_predictions(RES / "predictions" / "stgat_seed0.npz")
    pred["stgat"] = st
    ref = st

    # 1. interactive maps at a programmatically chosen AM-peak anchor
    s = cm.pick_am_peak_anchor(ref.target_ts, ref.y_true, ref.mask)
    times = {h: str(pd.Timestamp(ref.target_ts[s, h - 1])) for h in (3, 6, 12)}
    (MAPS).mkdir(parents=True, exist_ok=True)
    (MAPS / "selected_timestamps.json").write_text(
        json.dumps({"anchor_index": s, "target_times": times}, indent=2)
    )
    layers = {
        "truth": {h: ref.y_true[s, h - 1] for h in (3, 6, 12)},
        "stgat": {h: pred["stgat"].y_pred[s, h - 1] for h in (3, 6, 12)},
        "xgb_global": {h: pred["xgb_global"].y_pred[s, h - 1] for h in (3, 6, 12)},
    }
    cm.forecast_map(
        layers, times, loc, MAPS / "forecast_am_peak.html", "Forecasts vs truth (AM peak, test set)"
    )
    sl = slice(s, s + 12)  # 12 consecutive anchors = one hour of +15 min targets
    stamps = [str(pd.Timestamp(t)) for t in ref.target_ts[sl, 2]]
    cm.animated_map(
        np.where(ref.mask[sl, 2], ref.y_true[sl, 2], 0),
        stamps,
        loc,
        MAPS / "animated_peak_truth.html",
    )
    cm.animated_map(pred["stgat"].y_pred[sl, 2], stamps, loc, MAPS / "animated_peak_stgat.html")

    # 2. error-difference maps (XGB - STGAT), horizons 3 and 12, all and peak
    masks = slice_masks(ref, cfg)
    rows = []
    for h in (3, 12):
        for tag, sel in (("all", None), ("peak", masks["peak"])):
            d = per_sensor_mae(pred["xgb_global"], h, sel) - per_sensor_mae(pred["stgat"], h, sel)
            cm.static_scatter(
                loc,
                d,
                f"MAE(xgb_global) - MAE(stgat), +{h * 5} min, {tag}",
                FIG / f"error_diff_h{h}_{tag}.png",
                "RdBu",
                0.0,
                "mph (>0: STGAT better)",
            )
            rows += [(h, tag, sid, v) for sid, v in zip(p.data.sensor_ids, d, strict=True)]
    pd.DataFrame(rows, columns=["horizon", "slice", "sensor_id", "mae_xgb_minus_stgat"]).to_csv(
        MET / "error_diff_per_sensor.csv", index=False
    )

    # 3. space-time diagram on one corridor
    cor = hp.find_corridor(p.data.distances, p.data.adj, p.data.sensor_ids)
    hp.save_corridor(cor, MET / "corridor.json")
    ci = cor["indices"]
    ts0 = pd.DatetimeIndex(ref.target_ts[:, 5])  # +30 min targets
    day_ok = ts0.day_of_week < 5
    day = (
        pd.Series(
            ((ref.y_true[:, 5][:, ci] < 40) & ref.mask[:, 5][:, ci]).sum(1).astype(float)
            * ((ts0.hour >= 5) & (ts0.hour < 11) & day_ok),
            index=ts0,
        )
        .groupby(ts0.date)
        .sum()
        .idxmax()
    )
    sel = np.flatnonzero((ts0.date == day) & (ts0.hour >= 5) & (ts0.hour < 11))
    hp.space_time_diagram(
        {
            "truth": np.where(ref.mask[sel][:, 5][:, ci], ref.y_true[sel][:, 5][:, ci], 0),
            "STGAT +30 min": pred["stgat"].y_pred[sel][:, 5][:, ci],
            "XGB-global +30 min": pred["xgb_global"].y_pred[sel][:, 5][:, ci],
        },
        ts0[sel],
        cor["sensor_ids"],
        f"Corridor space-time diagram, {day} (forecast shown at target time)",
        FIG / "space_time.png",
    )

    # 4. attention: AM peak vs night, averaged over test weekdays
    ck = torch.load("checkpoints/dl/stgat_seed0.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"], p.data.adj, p.data.locations, len(p.data.sensor_ids))
    model.load_state_dict(ck["state"])
    w = p.windows["test"]
    anc_ts = pd.DatetimeIndex(p.index[w.anchors])
    wk = anc_ts.day_of_week < 5
    hm = anc_ts.hour * 60 + anc_ts.minute
    pos = {
        "am_peak": np.flatnonzero(wk & (hm >= 450) & (hm < 510)),
        "night": np.flatnonzero(wk & (hm >= 120) & (hm < 240)),
    }
    adj = p.data.adj
    att, shares = {}, []
    for k, ix in pos.items():
        att[k] = av.mean_attention(model, w, ix[::3], channel_slice(ck["cfg"]), torch.device("cpu"))
        shares.append({"period": k, "n_windows": len(ix[::3]), **av.attention_shares(att[k], adj)})
    pd.DataFrame(shares).to_csv(MET / "attention_summary.csv", index=False)
    cong = ((ref.y_true < 40) & ref.mask).mean((0, 1))
    targets = [int(i) for i in np.argsort(cong)[::-1][:3]]
    for k in pos:
        av.plot_attention_map(att[k], loc, targets, k, FIG / f"attention_{k}.png")

    # 5. summary figures from the evaluation tables
    agg = pd.read_csv(MET / "all_metrics_agg.csv")
    hp.error_vs_horizon_all(agg, RES / "figures" / "error_vs_horizon_all_models.png")
    po = pd.read_csv(MET / "peak_offpeak.csv")
    hp.peak_offpeak_bars(po, RES / "figures" / "peak_offpeak_bars.png")
    print("maps done; anchor", times, "corridor length", len(ci))


if __name__ == "__main__":
    main()
