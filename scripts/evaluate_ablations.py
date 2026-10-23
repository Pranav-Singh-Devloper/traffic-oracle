"""Aggregate ablation predictions into results/metrics/ablations.csv and a bar chart.

Models can have different numbers of test anchors (H=6/12/24 shift the first window), so all
runs are compared on the INTERSECTION of target timestamps.

Usage: python scripts/evaluate_ablations.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic.evaluation.io import Predictions, load_predictions
from traffic.evaluation.metrics import METRICS

HEAD = (3, 6, 12)


def subset(p: Predictions, keep: np.ndarray) -> Predictions:
    first = p.target_ts[:, 0]
    sel = np.isin(first, keep)
    return Predictions(
        p.name,
        p.seed,
        p.y_pred[sel],
        p.y_true[sel],
        p.mask[sel],
        p.target_ts[sel],
        p.sensor_ids,
        p.meta,
    )


def main() -> None:
    d = Path("results/ablations/predictions")
    files = sorted(d.glob("*.npz"))
    if not files:
        raise SystemExit("no ablation predictions")
    preds = [load_predictions(f) for f in files]
    common = preds[0].target_ts[:, 0]
    for p in preds[1:]:
        common = np.intersect1d(common, p.target_ts[:, 0])
    print(f"{len(preds)} runs, {len(common)} common test anchors")
    rows = []
    for p in map(lambda q: subset(q, common), preds):
        m = re.match(r"abl_(A\d+)_(.+)", p.name)
        aid, var = (m.group(1), m.group(2)) if m else ("A0", "default")
        for h in HEAD:
            for mn, fn in METRICS.items():
                rows.append(
                    (
                        aid,
                        var,
                        p.seed,
                        h * 5,
                        mn,
                        fn(p.y_true[:, h - 1], p.y_pred[:, h - 1], p.mask[:, h - 1]),
                        p.meta.get("n_params"),
                        p.meta.get("epochs"),
                        p.meta.get("train_seconds"),
                    )
                )
    df = pd.DataFrame(
        rows,
        columns=[
            "ablation",
            "variant",
            "seed",
            "horizon_min",
            "metric",
            "value",
            "n_params",
            "epochs",
            "train_seconds",
        ],
    )
    df.to_csv("results/metrics/ablations_long.csv", index=False)
    agg = (
        df.groupby(["ablation", "variant", "horizon_min", "metric"])
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            n_seeds=("seed", "nunique"),
            n_params=("n_params", "first"),
            epochs=("epochs", "mean"),
        )
        .reset_index()
    )
    agg.to_csv("results/metrics/ablations.csv", index=False)
    mae = (
        agg[agg.metric == "MAE"]
        .pivot_table(index=["ablation", "variant"], columns="horizon_min", values="mean")
        .reset_index()
    )
    ref = mae[mae.ablation == "A0"].iloc[0]
    mae["d15"], mae["d30"], mae["d60"] = (mae[15] - ref[15], mae[30] - ref[30], mae[60] - ref[60])
    print(mae.round(3).to_string(index=False))
    mae = mae[mae.ablation != "A0"].sort_values(["ablation", "variant"])
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(mae) + 4), 5))
    x = np.arange(len(mae))
    for k, (h, c) in enumerate(zip((15, 30, 60), ("#56B4E9", "#E69F00", "#D55E00"), strict=True)):
        ax.bar(x + (k - 1) * 0.27, mae[h], 0.27, label=f"MAE@{h}min", color=c)
        ax.axhline(ref[h], color=c, ls=":", lw=1)
    ax.set_xticks(
        x,
        [f"{a}\n{v}" for a, v in zip(mae.ablation, mae.variant, strict=True)],
        fontsize=7,
        rotation=60,
    )
    ax.set(
        ylabel="Test MAE (mph); dotted = default",
        title="STGAT ablations (reduced budget, one change each)",
    )
    ax.legend()
    fig.tight_layout()
    out = Path("results/figures/ablations")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "ablation_bars.png", dpi=150)
    fig.savefig(out / "ablation_bars.pdf")


if __name__ == "__main__":
    main()
