"""Evaluate every prediction file in results/predictions and write results/metrics/*.csv.

Usage: python scripts/evaluate_all.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from traffic.config import load_config
from traffic.evaluation.io import assert_aligned, load_predictions
from traffic.evaluation.slices import evaluate, per_sensor_mae
from traffic.evaluation.stats_tests import block_bootstrap_ci, day_blocks, diebold_mariano

HEADLINE = (3, 6, 12)


def aggregate(long: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds (std is NaN when there is a single run)."""
    g = long.groupby(["model", "slice", "horizon", "metric"], dropna=False)
    out = g["value"].agg(["mean", "std", "count"]).rename(columns={"count": "n_runs"}).reset_index()
    n = g["n_obs"].first().reset_index(drop=True)
    out["n_obs"] = n
    return out


def main() -> None:
    cfg = load_config("configs/base.yaml")
    pred_dir = Path(cfg["paths"]["results"]) / "predictions"
    met = Path(cfg["paths"]["results"]) / "metrics"
    met.mkdir(parents=True, exist_ok=True)
    files = sorted(pred_dir.glob("*.npz"))
    if not files:
        raise SystemExit("no prediction files found")
    preds = [load_predictions(f) for f in files]
    ref = next((p for p in preds if p.name == "persistence"), preds[0])
    for p in preds:
        assert_aligned(ref, p)  # hard error if misaligned

    long = pd.concat([evaluate(p, cfg) for p in preds], ignore_index=True)
    long.to_csv(met / "all_metrics_long.csv", index=False)
    agg = aggregate(long)
    agg.to_csv(met / "all_metrics_agg.csv", index=False)

    def wide(df: pd.DataFrame) -> pd.DataFrame:
        w = df.pivot_table(index=["model", "slice", "horizon"], columns="metric", values="mean")
        sd = df.pivot_table(index=["model", "slice", "horizon"], columns="metric", values="std")
        w.columns = [f"{c}" for c in w.columns]
        for c in sd.columns:
            w[f"{c}_std"] = sd[c]
        return w.reset_index()

    head = agg[agg.horizon.isin(HEADLINE)]
    w = wide(head)
    w["horizon_min"] = w["horizon"] * cfg["interval_minutes"]
    main_all = w[w.slice == "all"].drop(columns="slice")
    daily = main_all[main_all.model.str.endswith("_daily")]
    daily.to_csv(met / "daily_lag_experiment.csv", index=False)  # separate experiment (SPEC 7.2)
    main_t = main_all[~main_all.model.str.endswith("_daily")]
    main_t.to_csv(met / "main_table.csv", index=False)
    w[w.slice.isin(["peak", "peak_am", "peak_pm", "offpeak_weekday", "weekend"])].to_csv(
        met / "peak_offpeak.csv", index=False
    )
    w[w.slice.isin(["congested", "freeflow"])].to_csv(met / "congested_freeflow.csv", index=False)

    # Per-sensor MAE (seed-averaged) for maps
    rows = []
    for p in preds:
        for h in HEADLINE:
            v = per_sensor_mae(p, h)
            rows += [(p.name, p.seed, h, sid, x) for sid, x in zip(p.sensor_ids, v, strict=True)]
    ps = pd.DataFrame(rows, columns=["model", "seed", "horizon", "sensor_id", "mae"])
    ps.groupby(["model", "horizon", "sensor_id"], as_index=False)["mae"].mean().to_csv(
        met / "per_sensor_mae.csv", index=False
    )

    # Block-bootstrap CIs (first seed of each model)
    ci_rows = []
    seen = set()
    for p in preds:
        if p.name in seen:
            continue
        seen.add(p.name)
        for h in HEADLINE:
            e, c = day_blocks(
                p.target_ts[:, 0], p.y_true[:, h - 1], p.y_pred[:, h - 1], p.mask[:, h - 1]
            )
            lo, hi = block_bootstrap_ci(e, c)
            ci_rows.append((p.name, h, e.sum() / max(c.sum(), 1), lo, hi))
    pd.DataFrame(ci_rows, columns=["model", "horizon", "MAE", "ci_lo", "ci_hi"]).to_csv(
        met / "bootstrap_ci.csv", index=False
    )

    # Significance: per seed pair, report median DM stat and the max (most conservative) p
    by_name: dict[str, list] = {}
    for p in preds:
        by_name.setdefault(p.name, []).append(p)
    sig_rows = []
    for a, b in cfg.get("significance_pairs", []):
        if a not in by_name or b not in by_name:
            continue
        for h in HEADLINE:
            k = max(len(by_name[a]), len(by_name[b]))
            res = []
            for i in range(k):
                pa, pb = by_name[a][i % len(by_name[a])], by_name[b][i % len(by_name[b])]
                res.append(
                    diebold_mariano(
                        pa.y_true[:, h - 1],
                        pa.y_pred[:, h - 1],
                        pb.y_pred[:, h - 1],
                        pa.mask[:, h - 1],
                        h,
                    )
                )
            sig_rows.append(
                (a, b, h, float(np.median([r[0] for r in res])), max(r[1] for r in res), k)
            )
    pd.DataFrame(
        sig_rows, columns=["model_a", "model_b", "horizon", "dm_stat", "p_value", "n_seed_pairs"]
    ).to_csv(met / "significance.csv", index=False)

    # Sanity warnings (SPEC 8.6)
    msgs = []
    mt = main_t.set_index(["model", "horizon"])
    models = sorted({m for m, _ in mt.index})
    base = {"persistence", "historical_average", "seasonal_naive"}
    for m in models:
        if (m, 12) in mt.index and mt.loc[(m, 12), "MAE"] < 3.0:
            msgs.append(
                f"WARNING: {m} MAE@60min={mt.loc[(m, 12), 'MAE']:.2f} < 3.0: investigate leakage."
            )
        if (
            m not in base
            and (m, 3) in mt.index
            and mt.loc[(m, 3), "MAE"] >= mt.loc[("persistence", 3), "MAE"]
        ):
            msgs.append(f"WARNING: {m} does not beat persistence at 15 min: something is broken.")
    (met / "sanity_warnings.txt").write_text("\n".join(msgs) or "no warnings")
    for m in msgs:
        warnings.warn(m, stacklevel=1)
    print(main_t.round(3).to_string(index=False))
    print("\n".join(msgs) or "sanity checks: no warnings")


if __name__ == "__main__":
    main()
