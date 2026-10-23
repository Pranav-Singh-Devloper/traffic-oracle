"""Load, verify, rebuild graph, print split boundaries/stats, write results/metrics JSON.

Usage: python scripts/prepare_data.py [--debug]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic.config import load_config
from traffic.data.graph import build_adj_from_distances, graph_stats, verify_against_pkl
from traffic.data.missing import outage_stats
from traffic.data.pipeline import prepare
from traffic.data.split import boundaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = prepare(cfg, cfg["paths"]["raw"], debug=args.debug)
    out = Path(cfg["paths"]["results"]) / "metrics"
    out.mkdir(parents=True, exist_ok=True)

    b = boundaries(p.split, p.index)
    print("Split boundaries:")
    for k, v in b.items():
        print(f"  {k:5s} {v['start']} -> {v['end']}  ({v['n_steps']} steps)")
    (out / "split_boundaries.json").write_text(json.dumps(b, indent=2))
    print("Windows:", {k: len(w) for k, w in p.windows.items()})

    d = p.data
    chk = verify_against_pkl(build_adj_from_distances(d.distances, d.sensor_ids), d.adj)
    print("Graph rebuild vs adj_mx.pkl:", chk)
    gs = graph_stats(d.adj)
    gs["rebuild_check"] = chk
    (out / "graph_stats.json").write_text(json.dumps(gs, indent=2))
    print({k: v for k, v in gs.items() if "hist" not in k})

    ms = outage_stats(p.mask, p.index)
    (out / "missing_stats.json").write_text(json.dumps(ms, indent=2))
    print(f"Missing rate {ms['overall_missing_rate']:.4f}; scaler {p.scaler.params()}")


if __name__ == "__main__":
    main()
