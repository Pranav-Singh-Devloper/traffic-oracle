"""Run the simulated ramp-metering experiment (offline simulation only; see DISCLAIMER).

Needs results/metrics/corridor.json (make_maps.py) and STGAT test predictions.
Usage: python scripts/run_control.py [--model stgat_seed0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic.config import load_config
from traffic.control.run_experiment import (
    DISCLAIMER,
    END_H,
    POLICIES,
    WARMUP_H,
    DayInputs,
    run_policies,
    simulate,
    tune_fixed_rate,
)
from traffic.data.pipeline import prepare
from traffic.evaluation.io import load_predictions

N_CELLS = 12


def interp(x: np.ndarray) -> np.ndarray:
    """Fill NaNs (outages) per column by interpolation, then edge values."""
    return pd.DataFrame(x).interpolate(limit_direction="both").to_numpy()


def make_day(p, pred, day, sensor_idx: list[int], ci_idx: list[int]) -> DayInputs:
    ix = p.index
    sel = np.flatnonzero(
        (ix.date == day)
        & (ix.hour * 60 + ix.minute >= WARMUP_H * 60)
        & (ix.hour * 60 + ix.minute < END_H * 60)
    )
    v = np.where(p.mask[sel], p.speed[sel], np.nan)
    ci = np.clip((65 - np.nanmean(interp(v[:, ci_idx]), axis=1)) / 45, 0, 1)
    fc_pred = np.full((len(sel), len(sensor_idx)), np.nan)
    fc_true = np.full_like(fc_pred, np.nan)
    if pred is not None:
        anchor_ts = pd.DatetimeIndex(pred.target_ts[:, 0]) - pd.Timedelta(minutes=5)
        pos = {t: i for i, t in enumerate(anchor_ts)}
        for r, t in enumerate(ix[sel]):
            s = pos.get(t)
            if s is not None:
                fc_pred[r] = pred.y_pred[s, 2][sensor_idx]
                fc_true[r] = np.where(
                    pred.mask[s, 2][sensor_idx], pred.y_true[s, 2][sensor_idx], np.nan
                )
        fc_true = interp(fc_true)
        fc_pred = interp(fc_pred)
    return DayInputs(ci, fc_pred, fc_true, str(day))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stgat_seed0")
    args = ap.parse_args()
    cfg = load_config("configs/base.yaml")
    p = prepare(cfg, cfg["paths"]["raw"])
    pred = load_predictions(Path("results/predictions") / f"{args.model}.npz")
    cor = json.loads(Path("results/metrics/corridor.json").read_text())
    ci_idx = cor["indices"]
    sensor_idx = [ci_idx[int(round(k))] for k in np.linspace(0, len(ci_idx) - 1, N_CELLS)]

    def congestion(days, lo, hi):
        d = pd.Series(0.0, index=days)
        return d

    ix = p.index
    wk = ix.day_of_week < 5
    am = (ix.hour >= 5) & (ix.hour < 11)
    corr_cong = pd.Series(
        ((p.speed[:, ci_idx] < 40) & p.mask[:, ci_idx]).mean(1) * (wk & am), index=ix
    )
    daily = corr_cong.groupby(ix.date).sum()
    split = p.split
    val_days = sorted(set(ix[split.val].date))
    test_days = sorted(set(pd.DatetimeIndex(pred.target_ts[:, 0]).date))
    val_day = daily.loc[val_days].idxmax()
    # only days fully inside the test predictions (need all slots)
    full = [d for d in test_days if (pd.DatetimeIndex(pred.target_ts[:, 0]).date == d).sum() > 250]
    days = list(daily.loc[full].sort_values(ascending=False).index[:5])
    print("val day", val_day, "test days", days)
    fixed, grid = tune_fixed_rate(make_day(p, None, val_day, sensor_idx, ci_idx))
    print("fixed-time rate (tuned on val day):", fixed)
    inputs = [make_day(p, pred, d, sensor_idx, ci_idx) for d in days]
    res = run_policies(inputs, fixed)
    out = Path("results/metrics")
    res.to_csv(out / "control_results.csv", index=False)
    grid.to_csv(out / "control_fixed_rate_tuning.csv", index=False)
    cols = [
        "TTS_veh_h",
        "distance_veh_mi",
        "mainline_avg_speed_mph",
        "max_ramp_queue_veh",
        "congestion_cell_minutes",
    ]
    summ = res.groupby("policy")[cols].agg(["mean", "std"]).reindex(POLICIES)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(out / "control_summary.csv")
    print(DISCLAIMER)
    print(summ.round(2).T.to_string())

    # figures (disclaimer in the caption area)
    fig_dir = Path("results/figures/control")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    m = res.groupby("policy")["TTS_veh_h"].agg(["mean", "std"]).reindex(POLICIES)
    ax.bar(range(len(m)), m["mean"], yerr=m["std"], color="#56B4E9")
    ax.set_xticks(range(len(m)), [x.replace("_", "\n") for x in m.index], fontsize=8)
    ax.set(ylabel="Total time spent (veh·h)", title="Simulated ramp metering (5 test weekdays)")
    fig.text(0.5, 0.005, DISCLAIMER[:95] + "...", ha="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(fig_dir / "tts_by_policy.png", dpi=150)
    plt.close(fig)
    d0 = inputs[0]
    fig, axes = plt.subplots(1, len(POLICIES), figsize=(3.4 * len(POLICIES), 3.6), sharey=True)
    for a, pol in zip(axes, POLICIES, strict=True):
        r = simulate(pol, d0, fixed)
        a.imshow(
            r["_density"].T,
            aspect="auto",
            origin="lower",
            cmap="magma_r",
            extent=[5, 11, -0.5, N_CELLS - 0.5],
            vmin=0,
            vmax=300,
        )
        a.set_title(pol.replace("_", " "), fontsize=8)
        a.set_xlabel("hour")
    axes[0].set_ylabel("cell (downstream ->)")
    fig.suptitle(f"Density (veh/mi) {d0.date}. Offline simulation only.", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "density_heatmaps.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
