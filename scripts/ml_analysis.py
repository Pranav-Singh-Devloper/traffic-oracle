"""Phase 1 analysis: error curves, gap analysis, SHAP / gain importance, per-sensor map.

Run after scripts/evaluate_all.py. Usage: python scripts/ml_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from traffic.config import load_config
from traffic.evaluation.io import load_predictions
from traffic.evaluation.slices import slice_masks
from traffic.evaluation.stats_tests import day_blocks, diebold_mariano, paired_block_bootstrap

RES = Path("results")
FIG = RES / "figures" / "ml"
MET = RES / "metrics"
ML_MODELS = ["xgb_per_segment", "xgb_global", "rf_multioutput", "xgb_spatial"]
BASE = ["persistence", "historical_average", "seasonal_naive"]


def error_vs_horizon() -> None:
    agg = pd.read_csv(MET / "all_metrics_agg.csv")
    agg = agg[(agg.slice == "all") & (agg.metric == "MAE")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in BASE + ML_MODELS:
        d = agg[agg.model == m].sort_values("horizon")
        if d.empty:
            continue
        style = "--" if m in BASE else "-"
        ax.plot(d.horizon * 5, d["mean"], style, marker="o", ms=4, label=m)
    ax.set(
        xlabel="Horizon (minutes)",
        ylabel="Test MAE (mph)",
        title="Error vs horizon (Phase 1 models)",
    )
    ax.legend(fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / "error_vs_horizon.png", dpi=150)
    fig.savefig(FIG / "error_vs_horizon.pdf")
    plt.close(fig)


def gap_analysis(cfg: dict) -> pd.DataFrame:
    a = load_predictions(RES / "predictions" / "xgb_global.npz")
    b = load_predictions(RES / "predictions" / "xgb_spatial.npz")
    masks = slice_masks(a, cfg)
    rows = []
    for h in range(1, a.y_pred.shape[1] + 1):
        for sl in ("all", "peak", "offpeak_weekday", "weekend", "congested", "freeflow"):
            m = a.mask[:, h - 1] & masks[sl][:, h - 1]
            y = a.y_true[:, h - 1]
            e1, cnt = day_blocks(a.target_ts[:, 0], y, a.y_pred[:, h - 1], m)
            e2, _ = day_blocks(a.target_ts[:, 0], y, b.y_pred[:, h - 1], m)
            d, lo, hi = paired_block_bootstrap(e1, e2, cnt)
            dm, p = diebold_mariano(y, a.y_pred[:, h - 1], b.y_pred[:, h - 1], m, h)
            rows.append((h, sl, e1.sum() / cnt.sum(), e2.sum() / cnt.sum(), d, lo, hi, dm, p))
    df = pd.DataFrame(
        rows,
        columns=[
            "horizon",
            "slice",
            "mae_global",
            "mae_spatial",
            "delta_mae",
            "ci_lo",
            "ci_hi",
            "dm_stat",
            "p_value",
        ],
    )
    df.to_csv(MET / "gap_analysis.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for panel, sls in zip(
        ax, (("all", "peak", "offpeak_weekday", "weekend"), ("congested", "freeflow")), strict=True
    ):
        for sl in sls:
            d = df[df.slice == sl]
            panel.plot(d.horizon * 5, d.delta_mae, marker="o", ms=3, label=sl)
            panel.fill_between(d.horizon * 5, d.ci_lo, d.ci_hi, alpha=0.15)
        panel.axhline(0, color="k", lw=0.8)
        panel.set(xlabel="Horizon (minutes)")
        panel.legend(fontsize=8)
    ax[0].set_ylabel("MAE(xgb_global) - MAE(xgb_spatial)  (mph, >0: spatial better)")
    fig.suptitle(
        "Gap analysis: what do hand-made neighbor features add? (95% paired block-bootstrap CI)"
    )
    fig.tight_layout()
    fig.savefig(FIG / "gap_analysis.png", dpi=150)
    fig.savefig(FIG / "gap_analysis.pdf")
    plt.close(fig)
    return df


def importance_and_shap(cfg: dict) -> None:
    from traffic.data.pipeline import prepare
    from traffic.features.build_tabular import build_tabular, targets

    p = prepare(cfg, cfg["paths"]["raw"])
    rng = np.random.default_rng(cfg["seed"])
    out_rows = []
    for name, spatial in (("xgb_global", False), ("xgb_spatial", True)):
        tab = build_tabular(p, "test", cfg, "global", spatial)
        for h in (3, 12):
            path = Path("checkpoints/ml") / f"{name}_h{h}.json"
            if not path.exists():
                continue
            m = xgb.XGBRegressor()
            m.load_model(path)
            bst = m.get_booster()
            gain = bst.get_score(importance_type="gain")
            g = pd.Series({tab.names[int(k[1:])]: v for k, v in gain.items()}).sort_values(
                ascending=False
            )
            for k, v in g.head(25).items():
                out_rows.append((name, h, "gain", k, v))
            fig, ax = plt.subplots(figsize=(7, 6))
            g.head(20)[::-1].plot.barh(ax=ax)
            ax.set(title=f"{name} h={h * 5}min: gain importance (top 20)", xlabel="gain")
            fig.tight_layout()
            fig.savefig(FIG / f"gain_{name}_h{h}.png", dpi=150)
            plt.close(fig)
            # TreeSHAP on a seeded sample of 5000 test rows with observed targets
            y, mk = targets(p, "test", h)
            idx = rng.choice(np.flatnonzero(mk.ravel()), 5000, replace=False)
            x = tab.flat()[idx]
            d = xgb.DMatrix(x, feature_names=None, feature_types=m.get_booster().feature_types)
            contrib = bst.predict(d, pred_contribs=True)[:, :-1]
            mean_abs = pd.Series(np.abs(contrib).mean(0), index=tab.names).sort_values(
                ascending=False
            )
            for k, v in mean_abs.head(25).items():
                out_rows.append((name, h, "mean_abs_shap", k, v))
            import shap

            plt.figure()
            shap.summary_plot(contrib, x, feature_names=tab.names, max_display=15, show=False)
            plt.title(f"{name} h={h * 5}min SHAP (5000 test rows)")
            plt.tight_layout()
            plt.savefig(FIG / f"shap_{name}_h{h}.png", dpi=150)
            plt.close("all")
    pd.DataFrame(out_rows, columns=["model", "horizon", "kind", "feature", "value"]).to_csv(
        MET / "feature_importance.csv", index=False
    )


def sensor_map(cfg: dict) -> None:
    from traffic.data.load import load_locations

    loc = load_locations(Path(cfg["paths"]["raw"]) / "graph_sensor_locations.csv")
    ps = pd.read_csv(MET / "per_sensor_mae.csv", dtype={"sensor_id": str})
    ps = ps[(ps.model == "xgb_global") & (ps.horizon == 12)].merge(loc, on="sensor_id")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(
        ps.longitude, ps.latitude, c=ps.mae, cmap="viridis", s=35, edgecolor="k", lw=0.3
    )
    ax.set_aspect(1 / np.cos(np.radians(ps.latitude.mean())))
    ax.set(title="xgb_global per-sensor MAE at 60 min", xlabel="Longitude", ylabel="Latitude")
    fig.colorbar(sc, label="MAE (mph)")
    fig.tight_layout()
    fig.savefig(FIG / "per_sensor_mae_xgb_global_h12.png", dpi=150)
    plt.close(fig)
    ps.sort_values("mae", ascending=False).head(10).to_csv(
        MET / "hardest_sensors_xgb_global_h12.csv", index=False
    )


def main() -> None:
    cfg = load_config("configs/base.yaml")
    FIG.mkdir(parents=True, exist_ok=True)
    error_vs_horizon()
    print(gap_analysis(cfg).query("horizon in (3, 6, 12)").round(3).to_string(index=False))
    sensor_map(cfg)
    importance_and_shap(cfg)


if __name__ == "__main__":
    main()
