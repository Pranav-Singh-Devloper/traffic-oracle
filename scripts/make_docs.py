"""Generate data-driven docs (hybrid, control, ablations) from results/metrics CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from traffic.evaluation.report_tables import to_markdown

MET = Path("results/metrics")
DISC = (
    "> This is an offline simulation on a synthetic network. It demonstrates how forecasts could "
    "inform a control policy. No live or real-world traffic control is performed or claimed.\n"
)


def control() -> None:
    s = pd.read_csv(MET / "control_summary.csv")
    cols = ["policy"] + [c for c in s.columns if c.endswith("_mean")]
    t = s[cols].copy()
    t.columns = [c.replace("_mean", "") for c in t.columns]
    txt = f"""# Simulated control results (auto-generated)

{DISC}
Assumptions: [control_assumptions.md](control_assumptions.md). Mean over 5 test weekdays; standard deviations are in `results/metrics/control_summary.csv`.

{to_markdown(t, 1)}

Reading guide: lower TTS, queue and congested cell-minutes are better; higher mainline speed is better.

Figures: `results/figures/control/`.
"""
    Path("docs/control_results.md").write_text(txt)


def gap() -> None:
    g = pd.read_csv(MET / "gap_analysis.csv")
    g = g[g.horizon.isin([3, 6, 12])][
        ["horizon", "slice", "mae_global", "mae_spatial", "delta_mae", "ci_lo", "ci_hi", "p_value"]
    ]
    dl = (
        pd.read_csv(MET / "daily_lag_experiment.csv")
        if (MET / "daily_lag_experiment.csv").exists()
        else None
    )
    txt = f"""# Gap analysis: do neighbour features help? (auto-generated)

`delta_mae = MAE(xgb_global) - MAE(xgb_spatial)`; positive means the hand-made spatial features help.
95% intervals are paired day-block bootstrap; `p_value` is Diebold-Mariano (HAC).

{to_markdown(g, 3)}

**Reading.** Spatial features help in almost every slice, most at peaks and in congested states, and the effect does not grow with the horizon (it is largest around 30 minutes). That supports "neighbours help during congestion" but only partially supports "graph benefit grows with horizon".
"""
    if dl is not None:
        txt += (
            "\n## Daily-lag side experiment (not in the main table)\n\n`lag_288` (same time yesterday) added to xgb_global:\n\n"
            + to_markdown(dl, 3)
            + "\n\nCompare with `xgb_global` in `results/metrics/main_table.csv`; the daily lag adds essentially nothing once the 12-step history, time-of-day features and sensor identity are present.\n"
        )
    Path("docs/gap_analysis.md").write_text(txt)


def hybrid() -> None:
    g = pd.read_csv(MET / "hybrid_gain.csv")
    g = g[
        (g.hybrid == "hybrid_residual")
        & g.component.isin(["xgb_global", "stgat"])
        & g.slice.isin(["all", "peak", "offpeak_weekday", "weekend", "congested", "freeflow"])
    ]
    g = g.pivot_table(
        index=["component", "slice"], columns="horizon_min", values="gain_mph"
    ).reset_index()
    gw = pd.read_csv(MET / "gate_weights_summary.csv")
    txt = f"""# Hybrid analysis (auto-generated)

## Gain of the residual hybrid over its components (mph, positive = hybrid better)
{to_markdown(g, 3)}

## Learned gate: mean weight on STGAT (test set)
{to_markdown(gw, 3)}

**Reading.** Over XGBoost-global the residual hybrid's gain grows with horizon and is largest in congested states and at peaks, where per-sensor history is least informative; in free flow it is slightly negative at 60 minutes. Over STGAT the gain is largest at 15 minutes and shrinks with horizon: STGAT already captures much of the local structure at long range. The gate puts more weight on STGAT at peaks and at longer horizons, as hypothesised. Significance (DM) is in `results/metrics/significance.csv`. Caveat: the gate used the last part of validation for early stopping (ADR-008).
"""
    Path("docs/hybrid_analysis.md").write_text(txt)


def ablations() -> None:
    a = pd.read_csv(MET / "ablations.csv")
    a = (
        a[a.metric == "MAE"]
        .pivot_table(index=["ablation", "variant"], columns="horizon_min", values="mean")
        .reset_index()
    )
    ref = a[a.ablation == "A0"].iloc[0]
    for h in (15, 30, 60):
        a[f"d{h}"] = a[h] - ref[h]
    txt = f"""# Ablation analysis (auto-generated table; reduced budget)

One seed, 12 epochs, same budget for the default (A0). Differences below about 0.05 mph at 15/30 minutes are within seed noise of the full runs.

{to_markdown(a, 3)}

**Reading.** See the Ablations section of `report/phase2/main.tex`. In short: under this budget the road graph is not shown to beat the identity graph (A1), a random graph is worse than both (A2), the node embedding (A7) and calendar features (A8) matter most, and everything else is within noise. A10 improving without regularisation reflects under-training at 12 epochs. A13 (hybrid design) is the residual/gate/average comparison in the main table.
"""
    Path("docs/ablation_analysis.md").write_text(txt)


def main() -> None:
    for name, fn, need in (
        ("gap", gap, "gap_analysis.csv"),
        ("hybrid", hybrid, "hybrid_gain.csv"),
        ("ablations", ablations, "ablations.csv"),
    ):
        if (MET / need).exists():
            fn()
            print("wrote docs for", name)
    if (MET / "control_summary.csv").exists():
        control()
        print("wrote docs/control_results.md")


if __name__ == "__main__":
    main()
