"""Error-vs-horizon, peak/off-peak bars and the space-time diagram."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = [
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


def error_vs_horizon_all(agg: pd.DataFrame, out: Path) -> None:
    """Key figure: MAE vs horizon for every model; DL/hybrids show +-std bands."""
    d = agg[(agg.slice == "all") & (agg.metric == "MAE")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab20")
    for i, m in enumerate([m for m in ORDER if m in set(d.model)]):
        x = d[d.model == m].sort_values("horizon")
        style = "--" if m in ORDER[:3] else "-"
        ax.plot(x.horizon * 5, x["mean"], style, marker="o", ms=3, color=cmap(i), label=m)
        if x["std"].notna().any():
            ax.fill_between(
                x.horizon * 5, x["mean"] - x["std"], x["mean"] + x["std"], color=cmap(i), alpha=0.2
            )
    ax.set(
        xlabel="Horizon (minutes)", ylabel="Test MAE (mph)", title="Error vs horizon, all models"
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def peak_offpeak_bars(po: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    models = [m for m in ORDER if m in set(po.model)]
    for ax, h in zip(axes, (3, 6, 12), strict=True):
        d = (
            po[po.horizon == h]
            .pivot_table(index="model", columns="slice", values="MAE")
            .reindex(models)
        )
        d[["peak", "offpeak_weekday", "weekend"]].plot.bar(ax=ax, legend=(h == 3))
        ax.set_title(f"MAE at {h * 5} min")
        ax.set_xlabel("")
    axes[0].set_ylabel("MAE (mph)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def find_corridor(
    dist_df: pd.DataFrame,
    w: np.ndarray,
    sensor_ids: list[str],
    min_len: int = 10,
    max_len: int = 25,
) -> dict:
    """Ordered chain of sensors along one direction of travel.

    Walk from every start along the nearest not-yet-visited downstream neighbour (W[i, j] > 0)
    while the road distance from the start keeps increasing; keep the longest chain.
    """
    n = len(sensor_ids)
    idx = {s: i for i, s in enumerate(sensor_ids)}
    dmat = np.full((n, n), np.inf)
    for f, t, c in dist_df[["from", "to", "cost"]].itertuples(index=False):
        if f in idx and t in idx:
            dmat[idx[f], idx[t]] = c
    best: list[int] = []
    for s in range(n):
        chain, seen = [s], {s}
        while len(chain) < max_len:
            cur = chain[-1]
            cand = [
                j for j in np.flatnonzero(w[cur] > 0) if j not in seen and dmat[s, j] > dmat[s, cur]
            ]
            if not cand:
                break
            nxt = min(cand, key=lambda j: dmat[cur, j])
            chain.append(int(nxt))
            seen.add(int(nxt))
        if len(chain) > len(best):
            best = chain
    return {
        "indices": best,
        "sensor_ids": [sensor_ids[i] for i in best],
        "distance_from_start_m": [float(dmat[best[0], i]) for i in best],
        "min_len_ok": len(best) >= min_len,
    }


def space_time_diagram(
    series: dict[str, np.ndarray],
    times: pd.DatetimeIndex,
    corridor_ids: list[str],
    title: str,
    out: Path,
) -> None:
    """series: panel name -> [T, L] speeds (sensors in travel order)."""
    fig, axes = plt.subplots(1, len(series), figsize=(5 * len(series), 4.5), sharey=True)
    for ax, (name, arr) in zip(np.atleast_1d(axes), series.items(), strict=True):
        im = ax.imshow(
            np.ma.masked_less_equal(arr, 0).T,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=10,
            vmax=70,
            extent=[0, len(times), -0.5, arr.shape[1] - 0.5],
        )
        ticks = np.arange(0, len(times), 12)
        ax.set_xticks(ticks, [times[t].strftime("%H:%M") for t in ticks], rotation=45)
        ax.set_title(name)
    np.atleast_1d(axes)[0].set_ylabel("Sensor index along travel direction")
    fig.colorbar(im, ax=axes, label="speed (mph)", shrink=0.85)
    fig.suptitle(title)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_corridor(c: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c, indent=2))
