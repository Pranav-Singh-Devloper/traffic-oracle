"""Generate all EDA figures + results/metrics/eda_stats.json.

Usage: python scripts/run_eda.py
"""

from __future__ import annotations

import json
from pathlib import Path

from traffic.config import load_config
from traffic.data.pipeline import prepare
from traffic.viz import eda_plots as eda


def main() -> None:
    cfg = load_config("configs/base.yaml")
    p = prepare(cfg, cfg["paths"]["raw"])
    fig = Path(cfg["paths"]["results"]) / "figures" / "eda"
    maps = Path(cfg["paths"]["results"]) / "maps"
    met = Path(cfg["paths"]["results"]) / "metrics"
    met.mkdir(parents=True, exist_ok=True)
    sp, mk, ix, d, ts = p.speed, p.mask, p.index, p.data, p.split.train.stop
    stats: dict = {}
    stats["missingness"] = eda.plot_missingness(mk, ix, fig)
    stats["daily_profile"] = eda.plot_daily_profile(sp, mk, ix, cfg, fig)
    stats["heatmap"] = eda.plot_tod_dow_heatmap(sp, mk, ix, fig)
    stats["distribution"] = eda.plot_speed_distribution(sp, mk, ix, cfg, fig)
    stats["acf"] = eda.plot_acf_pacf(sp, mk, ix, fig, ts)
    stats["corr_vs_hops"] = eda.plot_corr_vs_hops(sp, mk, d.adj, fig, ts)
    stats["xcorr"] = eda.plot_lagged_xcorr(d.adj, sp, mk, ix, cfg, fig, met, ts)
    stats["sensor_map"] = eda.sensor_map(sp, mk, d.locations, maps / "eda_sensors.html")
    stats["congestion_map"] = eda.congestion_frequency_map(
        sp, mk, ix, d.locations, cfg, fig, maps / "eda_congestion_frequency.html"
    )
    stats["splits"] = eda.plot_split_timeline(sp, mk, ix, p.split, cfg, fig)
    (met / "eda_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2)[:6000])


if __name__ == "__main__":
    main()
