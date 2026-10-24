"""Generate report/tables/numbers.tex: one \\newcommand per number quoted in the prose.

LaTeX macro names cannot contain digits, so horizons are spelled out (Fifteen/Thirty/Sixty)
and model names are CamelCased (xgb_global -> XgbGlobal).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

WORD = {3: "Fifteen", 6: "Thirty", 12: "Sixty"}
SLICE = {
    "peak": "Peak",
    "peak_am": "PeakAm",
    "peak_pm": "PeakPm",
    "offpeak_weekday": "Offpeak",
    "weekend": "Weekend",
    "congested": "Congested",
    "freeflow": "Freeflow",
    "all": "All",
}


def camel(s: str) -> str:
    return "".join(p.capitalize() for p in s.replace("-", "_").split("_"))


DIGITS = dict(
    zip(
        "0123456789",
        ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"],
        strict=True,
    )
)


def nodigits(s: str) -> str:
    """LaTeX control-sequence names may not contain digits."""
    return "".join(DIGITS.get(c, c) for c in s)


def _fmt(v, d: int = 2) -> str:
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def build_macros(met: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    mt = pd.read_csv(met / "main_table.csv")
    for r in mt.itertuples():
        if r.horizon not in WORD:
            continue
        key = f"{camel(r.model)}{WORD[r.horizon]}"
        m[f"mae{key}"], m[f"rmse{key}"], m[f"mape{key}"] = (
            _fmt(r.MAE),
            _fmt(r.RMSE),
            _fmt(r.MAPE, 1),
        )
        for c in ("MAE", "RMSE"):
            sd = getattr(r, f"{c}_std", np.nan)
            if not pd.isna(sd):
                m[f"{c.lower()}{key}Std"] = _fmt(sd)
    po = met / "peak_offpeak.csv"
    if po.exists():
        for r in pd.read_csv(po).itertuples():
            if r.horizon in WORD:
                m[f"mae{camel(r.model)}{WORD[r.horizon]}{SLICE[r.slice]}"] = _fmt(r.MAE)
    cf = met / "congested_freeflow.csv"
    if cf.exists():
        for r in pd.read_csv(cf).itertuples():
            if r.horizon in WORD:
                m[f"mae{camel(r.model)}{WORD[r.horizon]}{SLICE[r.slice]}"] = _fmt(r.MAE)
    ga = met / "gap_analysis.csv"
    if ga.exists():
        for r in pd.read_csv(ga).itertuples():
            if r.horizon in WORD and r.slice in SLICE:
                k = f"gap{SLICE[r.slice]}{WORD[r.horizon]}"
                m[k], m[k + "Lo"], m[k + "Hi"] = (
                    _fmt(r.delta_mae, 3),
                    _fmt(r.ci_lo, 3),
                    _fmt(r.ci_hi, 3),
                )
    hg = met / "hybrid_gain.csv"
    if hg.exists():
        for r in pd.read_csv(hg).itertuples():
            wh = {15: "Fifteen", 30: "Thirty", 60: "Sixty"}.get(r.horizon_min)
            if (
                wh
                and r.slice in SLICE
                and r.hybrid == "hybrid_residual"
                and r.component in ("xgb_global", "stgat")
            ):
                m[f"gain{camel(r.component)}{SLICE[r.slice]}{wh}"] = _fmt(r.gain_mph, 3)
    gw = met / "gate_weights_summary.csv"
    if gw.exists():
        for r in pd.read_csv(gw).itertuples():
            if r.horizon in (3, 6, 12):
                m[f"gatePeak{WORD[r.horizon]}"] = _fmt(r.mean_w_stgat_peak)
                m[f"gateOffpeak{WORD[r.horizon]}"] = _fmt(r.mean_w_stgat_offpeak)
    ab = met / "ablations.csv"
    if ab.exists():
        a = pd.read_csv(ab)
        a = a[a.metric == "MAE"]
        for r in a.itertuples():
            wh = {15: "Fifteen", 30: "Thirty", 60: "Sixty"}.get(r.horizon_min)
            if wh:
                m[f"abl{nodigits(camel(r.ablation) + camel(r.variant))}{wh}"] = _fmt(r.mean)
    eda = met / "eda_stats.json"
    if eda.exists():
        s = json.loads(eda.read_text())
        m["missingRate"] = _fmt(100 * s["missingness"]["missing_rate_overall"], 1)
        m["networkOutages"] = str(s["missingness"]["whole_network_outage_timestamps"])
        d = s["distribution"]
        m["congPeak"] = _fmt(100 * d["share_below_threshold_peak"], 1)
        m["congOffpeak"] = _fmt(100 * d["share_below_threshold_offpeak"], 1)
        m["meanSpeed"] = _fmt(d["mean_speed"], 1)
        c = s["corr_vs_hops"]
        m["corrOneHop"], m["corrTwoHop"] = _fmt(c["median_corr_1_hops"]), _fmt(
            c["median_corr_2_hops"]
        )
        m["corrEightHop"] = _fmt(c["median_corr_8_hops"])
        x = s["xcorr"]
        m["xcorrPairs"] = str(x["xcorr_pairs"])
        for k, v in (("Pos", "positive"), ("Zero", "zero"), ("Neg", "negative")):
            m[f"xcorrAll{k}"] = _fmt(100 * x[f"xcorr_share_peak_lag_{v}"], 0)
            m[f"xcorrPeak{k}"] = _fmt(100 * x[f"xcorr_peakhours_share_{v}"], 0)
        for sp, sd in s["splits"]["split_stats"].items():
            m[f"missing{camel(sp)}"] = _fmt(100 * sd["missing_rate"], 1)
    gs = met / "graph_stats.json"
    if gs.exists():
        g = json.loads(gs.read_text())
        m["graphEdges"], m["graphNodes"] = str(g["edges"]), str(g["nodes"])
        m["graphDiameter"] = str(g["diameter_largest_scc"])
        m["graphSCC"] = str(g["largest_scc"])
    sb = met / "split_boundaries.json"
    if sb.exists():
        b = json.loads(sb.read_text())
        for k, v in b.items():
            m[f"split{camel(k)}Start"] = v["start"][:10]
            m[f"split{camel(k)}End"] = v["end"][:10]
    for fname, prefix in (("control_summary.csv", "ctl"),):
        p = met / fname
        if p.exists():
            df = pd.read_csv(p).set_index("policy")
            for pol, r in df.iterrows():
                for col in ("TTS_veh_h", "max_ramp_queue_veh", "congestion_cell_minutes"):
                    m[f"{prefix}{camel(pol)}{camel(col.split('_')[0])}"] = _fmt(r[f"{col}_mean"], 1)
    for r in (met / "attention_summary.csv",):
        if r.exists():
            for row in pd.read_csv(r).itertuples():
                for k in ("self", "downstream_only", "upstream_only", "both_directions"):
                    m[f"att{camel(row.period)}{camel(k)}"] = _fmt(100 * getattr(row, k), 1)
    return m


def write_numbers(met: Path, out: Path) -> int:
    macros = build_macros(met)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% AUTO-GENERATED by scripts/make_report_assets.py. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    out.write_text("\n".join(lines) + "\n")
    return len(macros)
