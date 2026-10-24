"""Where do the hybrid gains come from? Writes results/metrics/hybrid_gain.csv + figure + docs.

Gain = MAE(component) - MAE(hybrid) (positive: hybrid better), seed-averaged, per slice/horizon.
Usage: python scripts/hybrid_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MET = Path("results/metrics")
HYB = ["hybrid_residual", "hybrid_gating", "hybrid_average"]
COMP = ["xgb_global", "stgat"]
SL = ["all", "peak", "offpeak_weekday", "weekend", "congested", "freeflow"]


def main() -> None:
    agg = pd.read_csv(MET / "all_metrics_agg.csv")
    a = agg[(agg.metric == "MAE") & agg.slice.isin(SL) & agg.horizon.isin([3, 6, 12])]
    mae = a.pivot_table(index=["slice", "horizon"], columns="model", values="mean")
    rows = []
    for h in HYB:
        for c in COMP + ["xgb_spatial", "gru_nograph"]:
            for (sl, hz), r in mae.iterrows():
                rows.append((h, c, sl, hz * 5, r[c], r[h], r[c] - r[h]))
    df = pd.DataFrame(
        rows,
        columns=[
            "hybrid",
            "component",
            "slice",
            "horizon_min",
            "mae_component",
            "mae_hybrid",
            "gain_mph",
        ],
    )
    df.to_csv(MET / "hybrid_gain.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, comp in zip(axes, COMP, strict=True):
        d = df[(df.component == comp) & (df.hybrid == "hybrid_residual") & df.slice.isin(SL)]
        p = d.pivot(index="slice", columns="horizon_min", values="gain_mph").reindex(SL)
        p.plot.bar(ax=ax)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(title=f"hybrid_residual gain over {comp} (mph)", xlabel="")
    fig.tight_layout()
    out = Path("results/figures/hybrid")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "hybrid_gain.png", dpi=150)
    fig.savefig(out / "hybrid_gain.pdf")
    gw = MET / "gate_weights.csv"
    if gw.exists():
        g = pd.read_csv(gw).groupby("horizon")[["mean_w_stgat_peak", "mean_w_stgat_offpeak"]].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(g.index * 5, g.mean_w_stgat_peak, marker="o", label="peak")
        ax.plot(g.index * 5, g.mean_w_stgat_offpeak, marker="o", label="off-peak")
        ax.set(
            xlabel="Horizon (min)",
            ylabel="mean gate weight on STGAT",
            title="Learned gate (test set)",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "gate_weights.png", dpi=150)
        fig.savefig(out / "gate_weights.pdf")
        g.to_csv(MET / "gate_weights_summary.csv")
    tab = df[(df.hybrid == "hybrid_residual") & (df.slice == "all")][
        ["component", "horizon_min", "gain_mph"]
    ]
    print(tab.round(3).to_string(index=False))
    print(
        df[
            (df.hybrid == "hybrid_residual")
            & (df.component == "xgb_global")
            & (df.horizon_min == 60)
        ]
        .round(3)
        .to_string(index=False)
    )
    if gw.exists():
        print(g.round(3))


if __name__ == "__main__":
    main()
