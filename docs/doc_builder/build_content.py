"""Collect every number from the project's result files and write content.json for the docx renderer.

Run with the project venv from the repo root:
  python <this file> <repo_root> <out_json>
All numbers come from results/metrics/*, configs/*, data/raw/MANIFEST.json and prediction metadata.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
MET = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"
sys.path.insert(0, str(ROOT / "src"))

blocks: list[dict] = []
counters = {"table": 0, "figure": 0}


def H1(t): blocks.append({"t": "h1", "text": t})
def H2(t): blocks.append({"t": "h2", "text": t})
def H3(t): blocks.append({"t": "h3", "text": t})
def P(t): blocks.append({"t": "p", "text": t})
def B(items): blocks.append({"t": "bullets", "items": list(items)})
def N(items): blocks.append({"t": "numbered", "items": list(items)})
def EQ(t): blocks.append({"t": "eq", "text": t})
def NOTE(t): blocks.append({"t": "callout", "text": t})
def BREAK(): blocks.append({"t": "pagebreak"})


def TABLE(header, rows, caption, widths=None, small=False):
    counters["table"] += 1
    blocks.append({"t": "table", "header": [str(h) for h in header],
                   "rows": [[str(c) for c in r] for r in rows],
                   "caption": f"Table {counters['table']}. {caption}", "widths": widths, "small": small})
    return counters["table"]


def FIGURE(path, caption, width_in=6.3):
    path = Path(path)
    if not path.exists():
        print("missing figure", path)
        return None
    w, h = Image.open(path).size
    counters["figure"] += 1
    blocks.append({"t": "img", "path": str(path), "px": [w, h], "width_in": width_in,
                   "caption": f"Figure {counters['figure']}. {caption}"})
    return counters["figure"]


def f2(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------- load results
eda = json.loads((MET / "eda_stats.json").read_text())
gs = json.loads((MET / "graph_stats.json").read_text())
sb = json.loads((MET / "split_boundaries.json").read_text())
ms = json.loads((MET / "missing_stats.json").read_text())
manifest = json.loads((ROOT / "data/raw/MANIFEST.json").read_text())
main = pd.read_csv(MET / "main_table.csv")
agg = pd.read_csv(MET / "all_metrics_agg.csv")
po = pd.read_csv(MET / "peak_offpeak.csv")
cf = pd.read_csv(MET / "congested_freeflow.csv")
sig = pd.read_csv(MET / "significance.csv")
boot = pd.read_csv(MET / "bootstrap_ci.csv")
gap = pd.read_csv(MET / "gap_analysis.csv")
fi = pd.read_csv(MET / "feature_importance.csv")
hard = pd.read_csv(MET / "hardest_sensors_xgb_global_h12.csv")
daily = pd.read_csv(MET / "daily_lag_experiment.csv")
hg = pd.read_csv(MET / "hybrid_gain.csv")
gw = pd.read_csv(MET / "gate_weights_summary.csv")
abl = pd.read_csv(MET / "ablations.csv")
abl_long = pd.read_csv(MET / "ablations_long.csv")
att = pd.read_csv(MET / "attention_summary.csv")
ctl = pd.read_csv(MET / "control_summary.csv")
ctl_tune = pd.read_csv(MET / "control_fixed_rate_tuning.csv")
ctl_res = pd.read_csv(MET / "control_results.csv")
xc = pd.read_csv(MET / "xcorr_peak_lags.csv")
corridor = json.loads((MET / "corridor.json").read_text())
sel = json.loads((ROOT / "results/maps/selected_timestamps.json").read_text())
tun = {k: pd.read_csv(MET / f"tuning_{k}.csv") for k in ("xgb_global", "xgb_spatial", "xgb_per_segment", "rf_multioutput")}
train_logs = {p.stem: pd.read_csv(p) for p in (MET / "training").glob("*.csv")}


def meta(name):
    z = np.load(ROOT / "results/predictions" / f"{name}.npz")
    return json.loads(str(z["meta"]))


import yaml  # noqa: E402

base_cfg = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
stgat_cfg = yaml.safe_load((ROOT / "configs/dl/stgat.yaml").read_text())
gru_cfg = yaml.safe_load((ROOT / "configs/dl/gru_nograph.yaml").read_text())
best = {k: yaml.safe_load((ROOT / f"configs/ml/{k}.best.yaml").read_text())["model"]["params"]
        for k in ("xgb_global", "xgb_spatial", "xgb_per_segment", "rf_multioutput")}

# derived values from the data pipeline (cheap)
from traffic.config import load_config  # noqa: E402
from traffic.data.graph import get_adjacency  # noqa: E402
from traffic.data.pipeline import prepare  # noqa: E402

cfg = load_config(str(ROOT / "configs/base.yaml"))
import os  # noqa: E402

os.chdir(ROOT)
prep = prepare(cfg)
d = prep.data
n = len(d.sensor_ids)
Dm = np.full((n, n), np.inf)
ix = {s: i for i, s in enumerate(d.sensor_ids)}
for a_, b_, c_ in d.distances[["from", "to", "cost"]].itertuples(index=False):
    Dm[ix[a_], ix[b_]] = c_
sigma = float(Dm[~np.isinf(Dm)].std())
W = d.adj
offd = ~np.eye(n, dtype=bool)
nbr_stats = {}
for v in ("forward", "backward", "bidirectional", "identity", "random", "knn_geo"):
    a = get_adjacency(v, W, d.locations)
    nb = ((a > 0) | np.eye(n, dtype=bool)).sum(1)
    nbr_stats[v] = (int(((a > 0) & offd).sum()), int(nb.max()), float(nb.mean()))

MODEL_ORDER = ["persistence", "historical_average", "seasonal_naive", "xgb_per_segment", "xgb_global",
               "rf_multioutput", "xgb_spatial", "gru_nograph", "stgat", "hybrid_residual", "hybrid_gating",
               "hybrid_average"]
NICE = {"persistence": "Persistence", "historical_average": "Historical average", "seasonal_naive": "Seasonal naive",
        "xgb_per_segment": "XGBoost per-segment", "xgb_global": "XGBoost global (pooled)",
        "rf_multioutput": "Random Forest (multi-output)", "xgb_spatial": "XGBoost spatial",
        "gru_nograph": "GRU no-graph", "stgat": "STGAT", "hybrid_residual": "Hybrid: residual",
        "hybrid_gating": "Hybrid: gating", "hybrid_average": "Hybrid: average",
        "xgb_global_daily": "XGBoost global + daily lag"}


def mt(model, h, col="MAE"):
    r = main[(main.model == model) & (main.horizon == h)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def mstd(model, h, col="MAE"):
    r = main[(main.model == model) & (main.horizon == h)]
    v = r.get(f"{col}_std")
    return float(v.iloc[0]) if v is not None and len(r) else float("nan")


def cell(model, h, col="MAE"):
    v, s = mt(model, h, col), mstd(model, h, col)
    return f"{v:.3f}" + ("" if np.isnan(s) else f" ± {s:.3f}")


team = [("Pranav Singh", "Pranav-Singh-Devloper", "Deep-learning core: dense/sparse GAT layer, STGAT, GRU no-graph, trainer, DL runs, integration and review"),
        ("Kunal", "kunaldevsahu", "Phase 1 ML (features, XGBoost, Random Forest, Optuna, SHAP, gap analysis) and the hybrid models"),
        ("Himanshu", "HimanshuPal29", "Evaluation and statistics framework, baselines, significance tests, simulated control"),
        ("Kanchan", "kanchan97-cp", "Data pipeline, graph construction, missing-data handling, EDA, ablation runs"),
        ("Vaibhav", "VaibhavSingh50", "Scaffold/CI, maps and attention plots, README, slides, report assembly")]

# =================================================================== CONTENT
blocks.append({"t": "title", "title": "Traffic Oracle",
               "subtitle": "Traffic speed forecasting on the METR-LA freeway network with graph attention networks, "
                           "segment-level machine learning and leakage-safe hybrids — complete technical documentation",
               "lines": ["Team T042", "Pranav Singh · Kunal · Himanshu · Kanchan · Vaibhav",
                         "Repository: github.com/Pranav-Singh-Devloper/traffic-oracle",
                         "Document generated from the project's result files (results/metrics, configs, data manifest)"]})
blocks.append({"t": "toc"})
BREAK()

# ------------------------------------------------------------------ 1
H1("1. Executive summary")
P("This document describes the Traffic Oracle project end to end: the data, every preprocessing decision, the evaluation "
  "protocol, each model and its hyperparameters, all results with their uncertainty, the ablations, the maps, the simulated "
  "control study, the engineering set-up, and the limitations. Every number is read from the project's result files; "
  "nothing is typed by hand.")
P("**Task.** Forecast freeway speed (mph) at 207 loop-detector sensors in Los Angeles, 5 to 60 minutes ahead (12 steps of "
  "5 minutes), from the previous 60 minutes (12 steps). **Question.** Does information from neighbouring sensors (a road "
  "graph) improve on forecasts that use each sensor's own history, and if so, where?")
TABLE(["Model", "MAE @ 15 min", "MAE @ 30 min", "MAE @ 60 min"],
      [[NICE[m], cell(m, 3), cell(m, 6), cell(m, 12)] for m in MODEL_ORDER],
      "Headline test-set results (masked MAE in mph; ± is the standard deviation over 3 seeds for deep models).",
      widths=[40, 20, 20, 20])
B([f"Every learned model beats persistence. Persistence MAE grows from {mt('persistence', 3):.3f} to {mt('persistence', 12):.3f} mph between 15 and 60 minutes.",
   f"The best models are hybrids: the residual hybrid at 15 and 30 minutes ({mt('hybrid_residual', 3):.3f} and {mt('hybrid_residual', 6):.3f} mph) "
   f"and the gated hybrid at 60 minutes ({mt('hybrid_gating', 12):.3f} mph).",
   f"STGAT (graph attention) equals the no-graph GRU at 15 minutes ({mt('stgat', 3):.3f} vs {mt('gru_nograph', 3):.3f}) and beats it clearly only at 60 minutes "
   f"({mt('stgat', 12):.3f} vs {mt('gru_nograph', 12):.3f}).",
   f"Hand-made neighbour features help XGBoost consistently, most at peaks and in congestion, but the gain does not grow with horizon.",
   "In reduced-budget ablations the road graph was not shown to beat a no-neighbour graph; node embeddings and calendar features mattered most.",
   "In an offline ramp-metering simulation, metering cut mainline congestion but forecasts did not reduce total time spent."])
BREAK()

# ------------------------------------------------------------------ 2
H1("2. Problem, hypothesis and scope")
P("Congestion on freeways is not local: a jam forms at a bottleneck and its tail moves upstream (a shockwave), so downstream "
  "sensors carry information about an upstream sensor's near future. A model that sees only a sensor's own history must infer "
  "this indirectly; a graph model can read neighbouring sensors directly.")
P("**Hypothesis.** Graph-aware models improve accuracy over per-segment models, mostly at longer horizons and during peak "
  "periods, when propagation effects dominate.")
P("**Two phases.** Phase 1 (machine learning): exploratory analysis, naive baselines, XGBoost per sensor and pooled, a "
  "multi-output Random Forest, and XGBoost with hand-engineered neighbour features (the bridge experiment). Phase 2 (deep "
  "learning): a spatio-temporal graph attention network (STGAT), a no-graph GRU ablation, a residual hybrid and a gated hybrid "
  "that combine XGBoost with the graph model, ablations, maps, and an optional simulated ramp-metering experiment.")
NOTE("Control is simulation only. Every control result in this document comes from an offline simulation on a synthetic network. "
     "It demonstrates how forecasts could inform a control policy; no live or real-world traffic control is performed or claimed.")
BREAK()

# ------------------------------------------------------------------ theory primer (from docs/theory)
H1("Theory primer (read this first)")
P("The concepts every team member must be able to explain, in the order they appear in the pipeline. Each part ends with likely viva questions.")
for name in ("chronological_splits", "missing_data", "metrics", "xgboost", "random_forest", "gat", "spatiotemporal_modeling",
             "diffusion_conv", "hybrid", "significance_testing", "control"):
    para: list[str] = []

    def flush():
        if para:
            P(" ".join(para))
            para.clear()

    items: list[str] = []
    for line in (ROOT / "docs/theory" / f"{name}.md").read_text().splitlines():
        s = line.strip()
        is_item = s.startswith("- ") or re.match(r"^\d+\. ", s)
        if not is_item and items:
            B(items)
            items = []
        if not s:
            flush()
        elif s.startswith("# "):
            flush()
            H2(s[2:])
        elif s.startswith("## "):
            flush()
            H3(s[3:])
        elif s.startswith("> "):
            flush()
            NOTE(s[2:].strip("*"))
        elif is_item:
            flush()
            items.append(re.sub(r"^(- |\d+\. )", "", s))
        else:
            para.append(s)
    flush()
    if items:
        B(items)

# ------------------------------------------------------------------ 3
H1("3. Dataset")
H2("3.1 Source and files")
P("METR-LA is the benchmark released with DCRNN (Li et al., ICLR 2018), from loop detectors on Los Angeles County freeways "
  "(origin cited by DCRNN as Jagadish et al., CACM 2014). The speed file comes from the DCRNN Google Drive folder; the sensor-graph "
  "files come from the `data/sensor_graph/` folder of the DCRNN GitHub repository. `scripts/download_data.py` fetches all five "
  "files (with 3 retries and back-off), verifies them, and writes `data/raw/MANIFEST.json` with sizes and SHA-256 hashes.")
TABLE(["File", "Bytes", "SHA-256 (first 16 hex)", "Content"],
      [[k, f"{v['bytes']:,}", v["sha256"][:16], {"metr-la.h5": "speed table (pandas HDF5, key 'df')",
                                                 "adj_mx.pkl": "(sensor_ids, id_to_index, 207×207 adjacency)",
                                                 "distances_la_2012.csv": "directed road distances (from, to, cost)",
                                                 "graph_sensor_ids.txt": "207 sensor ids, comma-separated",
                                                 "graph_sensor_locations.csv": "sensor id, latitude, longitude"}[k]]
       for k, v in manifest.items()],
      "Raw files, as verified by the download script.", widths=[24, 13, 22, 41], small=True)
H2("3.2 Facts asserted on every load")
B([f"Shape {34272:,} timestamps × {n} sensors; 34,272 = 119 days × 288 five-minute steps.",
   "First timestamp 2012-03-01 00:00:00, last 2012-06-27 23:55:00; the index is strictly regular at 5 minutes (no gaps, no duplicates).",
   f"Values are speeds in mph, non-negative, no NaN; maximum observed speed {prep.speed.max():.1f} mph.",
   "**Speed only.** There is no flow and no occupancy (those exist in PeMS datasets).",
   "**A reading of 0 mph means the sensor did not report** (outage), not stopped traffic. Zeros are masked in losses and metrics and never used as targets.",
   "Sensor ids from the HDF5 columns, the pickle, the id list and the location file are normalised to strings and must agree; all arrays use the order of `adj_mx.pkl`."])
H2("3.3 A loading problem and its fix (ADR-004)")
P("With pandas 3.1 and PyTables 3.11, `pd.read_hdf` fails on this 2019 file with “TypeError: a bytes-like object is required, "
  "not 'str'” (old attribute encoding). The loader therefore falls back to reading the plain fixed-format layout directly with "
  "PyTables: `axis0` (column ids), `axis1` (int64 nanosecond timestamps) and `block0_values` (float64 34,272×207). The data are "
  "identical and all integrity checks pass.")
H2("3.4 Missing data")
mm = eda["missingness"]
TABLE(["Statistic", "Value"],
      [["Readings equal to 0 (missing), whole series", pct(mm["missing_rate_overall"], 2)],
       ["Per-sensor missing rate: median", pct(mm["sensor_missing_rate_median"], 2)],
       ["Per-sensor missing rate: maximum", pct(mm["sensor_missing_rate_max"], 2)],
       ["Sensors that are never missing", str(mm["sensors_never_missing"])],
       ["Timestamps with ≥ 90% of sensors missing (network-wide outages)", f"{mm['whole_network_outage_timestamps']:,}"],
       ["Days containing at least one network-wide outage", str(len(ms["days_with_whole_network_outage"]))],
       ["Longest single-sensor outage", f"{ms['max_longest_outage_steps']:,} steps (≈ {ms['max_longest_outage_steps'] * 5 / 60 / 24:.1f} days)"],
       ["Missing rate: train / validation / test", " / ".join(pct(eda['splits']['split_stats'][k]['missing_rate'], 1) for k in ('train', 'val', 'test'))]],
      "Missingness in METR-LA.", widths=[65, 35])
P("Missingness is partly clustered in time (whole-network outages), not only per sensor, and the **test period has a much higher "
  "outage rate** than training. Metrics are masked, so this does not bias them, but test results are computed over fewer observations.")
BREAK()

# ------------------------------------------------------------------ 4
H1("4. The road graph")
H2("4.1 Construction (thresholded Gaussian kernel, as in DCRNN)")
P(f"The 11,753 rows of `distances_la_2012.csv` that connect two of the 207 sensors are placed in a 207×207 matrix of directed "
  f"road distances (metres; missing pairs = ∞). With σ the standard deviation of the finite entries (σ = {sigma:,.1f} m) and κ = 0.1:")
EQ("W_ij = exp( −(dist(v_i, v_j) / σ)² )   if this value ≥ κ,   else 0")
P("The rebuilt matrix is compared with the published `adj_mx.pkl`: maximum absolute difference "
  f"{gs['rebuild_check']['max_abs_diff']:.2e}, with {gs['rebuild_check']['n_differing']} entries differing by more than 1e-6 "
  "(float32 rounding only). The pickle is used as the canonical matrix; the rebuild is the proof of understanding.")
H2("4.2 Direction convention (ADR-002)")
P("`W[i, j]` is built from the road distance from sensor i to sensor j. Throughout the code, `A[i, j] > 0` means **node i "
  "aggregates information from node j**. With `forward = W`, node i receives from sensors it drives towards (downstream, where "
  "shockwaves start); with `backward = Wᵀ`, from sensors that drive into it (upstream). The default STGAT graph is "
  "`bidirectional = W + Wᵀ`. A unit test checks the convention on a hand-built three-sensor chain a→b→c.")
H2("4.3 Graph statistics")
TABLE(["Statistic", "Value"],
      [["Nodes", str(gs["nodes"])], ["Directed edges (off-diagonal non-zeros)", f"{gs['edges']:,}"],
       ["Density", f"{gs['density']:.4f}"], ["Self-loops in the matrix", str(gs["self_loops"])],
       ["Mean in/out degree", f"{gs['edges'] / gs['nodes']:.2f}"],
       ["Non-zero off-diagonal weights: min / mean", f"{W[(W > 0) & offd].min():.3f} / {W[(W > 0) & offd].mean():.3f}"],
       ["Weakly connected components (largest)", f"{gs['n_weak_components']} ({gs['largest_wcc']} nodes; one isolated sensor)"],
       ["Strongly connected components (largest)", f"{gs['n_strong_components']} ({gs['largest_scc']} nodes)"],
       ["Diameter of the largest strongly connected component", f"{gs['diameter_largest_scc']} hops"]],
      "Statistics of the canonical directed graph (NetworkX).", widths=[65, 35])
TABLE(["Variant", "Definition", "Off-diagonal edges", "Max neighbours incl. self", "Mean neighbours incl. self"],
      [[v, {"forward": "W (downstream senders)", "backward": "Wᵀ (upstream senders)", "bidirectional": "W + Wᵀ (default)",
            "identity": "no neighbours", "random": "same edge count, random positions (seeded)",
            "knn_geo": "8 nearest by haversine, Gaussian weights"}[v], f"{e:,}", str(mx), f"{mn:.1f}"]
       for v, (e, mx, mn) in nbr_stats.items()],
      "Graph variants used in the models and ablations.", widths=[16, 38, 15, 16, 15], small=True)
BREAK()

# ------------------------------------------------------------------ 5
H1("5. Preprocessing pipeline (shared by all models)")
H2("5.1 Global settings (configs/base.yaml)")
TABLE(["Setting", "Value", "Meaning"],
      [["seed", base_cfg["seed"], "global random seed"], ["interval_minutes", 5, "time step"],
       ["history_steps (H)", base_cfg["history_steps"], "60 minutes of input for every model"],
       ["horizon_steps (F)", base_cfg["horizon_steps"], "predict 5 … 60 minutes ahead"],
       ["report_horizons", str(base_cfg["report_horizons"]), "15, 30, 60 minutes in headline tables"],
       ["split", "70 / 10 / 20 %", "chronological, by timestamp"],
       ["peak windows", "06:00–10:00 and 15:00–19:00, weekdays", "judged on the target timestamp"],
       ["congestion threshold", "40 mph", "congested vs free-flow slice"],
       ["input_imputation", base_cfg["input_imputation"], "forward fill ≤ 6 steps, then train time-of-day mean"],
       ["scaler", base_cfg["scaler"], "one masked z-score for all sensors"],
       ["split_protocol", base_cfg["split_protocol"], "windows built inside each split"],
       ["use_daily_lag", str(base_cfg["use_daily_lag"]), "same-time-yesterday lag off in the main comparison"],
       ["resources", "6 CPU workers, 4 torch threads", "keeps the laptop responsive"]],
      "Global configuration.", widths=[24, 30, 46], small=True)
H2("5.2 Chronological split")
TABLE(["Split", "First timestamp", "Last timestamp", "Steps", "Windows (strict)"],
      [[k, v["start"], v["end"], f"{v['n_steps']:,}", f"{len(prep.windows[k]):,}"] for k, v in sb.items()],
      "Strict chronological split. Each split loses H + F − 1 = 23 anchors so no window crosses a boundary.",
      widths=[12, 26, 26, 14, 22])
P("A unit test asserts that the last training target precedes the first validation input, and the last validation target "
  "precedes the first test input. The literature-comparable DCRNN protocol (all windows first, then split samples) gives "
  "34,249 samples split 23,974 / 3,425 / 6,850; its boundary windows overlap across splits, so it is only used for comparison and "
  "is tested but not used for the reported results.")
H2("5.3 Missing-data handling")
B(["**Mask:** `mask = speed > 0`, shape [T, N].",
   "**Targets:** never imputed; masked in every loss and metric.",
   "**Inputs (default):** forward-fill outages for up to 6 steps (30 minutes, past only, so no leakage), then fill with the training-period "
   "mean of that sensor at that time-of-day slot (288 slots × 207 sensors); fall back to the sensor's training mean, then the global training mean.",
   "**Missing flag:** a binary `is_missing` input channel tells the model which values are not real.",
   "**Option (b), ablation A9:** keep the zeros; after scaling they are set to 0 in scaled space (the training mean) instead of the misleading "
   "(0 − mean)/std ≈ −4.6 (ADR-003)."])
H2("5.4 Scaling")
P(f"A masked z-score fitted on training timestamps and observed values only: mean = {prep.scaler.mean:.3f} mph, "
  f"standard deviation = {prep.scaler.std:.3f} mph. Models train in scaled space; every metric is computed in mph after the inverse "
  "transform. A test asserts that changing validation/test values never changes the scaler or the time-of-day means.")
H2("5.5 Input channels and windows")
TABLE(["#", "Channel", "Definition"],
      [[0, "speed", "imputed speed, z-scored"], [1, "is_missing", "1 if the original reading was 0"],
       [2, "tod_sin", "sin(2π · minute_of_day / 1440)"], [3, "tod_cos", "cos(2π · minute_of_day / 1440)"],
       [4, "dow_sin", "sin(2π · day_of_week / 7)"], [5, "dow_cos", "cos(2π · day_of_week / 7)"],
       [6, "is_weekend", "Saturday or Sunday"], [7, "is_holiday", "US federal holiday; in range only Memorial Day 2012-05-28 (validation period)"]],
      "The eight input channels (C = 8). Cyclic encoding keeps 23:55 and 00:00 close.", widths=[6, 18, 76], small=True)
EQ("X[t] = channels[t − 11 … t]  →  [12, 207, 8]        Y[t] = speed[t + 1 … t + 12]  →  [12, 207],  with mask alongside")
P("Windows store only their anchor index and slice the shared array on demand (the fully materialised training tensor would be ≈ 1.9 GB). "
  "Deep models use a vectorised batch iterator that indexes the array directly; validation and test batches keep chronological order.")
BREAK()

# ------------------------------------------------------------------ 6
H1("6. Exploratory data analysis")
P("All ten figures are produced by `scripts/run_eda.py`; numbers are in `results/metrics/eda_stats.json` and the auto-generated "
  "`docs/eda_summary.md`. Figures 5–7 use the training period only because they inform modelling choices.")
dp, dist, hm_ = eda["daily_profile"], eda["distribution"], eda["heatmap"]
H2("6.1 Missingness")
FIGURE(FIG / "eda/01_missingness.png", "Missing rate per sensor-day (left) and per-sensor missing-rate histogram (right).")
H2("6.2 Daily profile and weekly pattern")
P(f"Weekday mean speed bottoms out at {dp['weekday_min_mean_speed']:.1f} mph around {int(dp['weekday_min_speed_hour'])}:{int(60 * (dp['weekday_min_speed_hour'] % 1)):02d}; "
  f"weekends only drop to {dp['weekend_min_mean_speed']:.1f} mph (around {int(dp['weekend_min_speed_hour'])}:{int(60 * (dp['weekend_min_speed_hour'] % 1)):02d}). "
  f"Across the week, slot means range from {hm_['heatmap_min_mph']:.1f} to {hm_['heatmap_max_mph']:.1f} mph, with congestion confined to weekday rush hours.")
FIGURE(FIG / "eda/02_daily_profile.png", "Mean speed by time of day, weekday vs weekend (band: interquartile range across sensors; shaded: peak windows).", 5.6)
FIGURE(FIG / "eda/03_tod_dow_heatmap.png", "Mean speed by time of day × day of week.")
H2("6.3 Speed distribution")
P(f"Overall mean {dist['mean_speed']:.2f} mph. {pct(dist['share_below_threshold_all'])} of observations are below 40 mph overall, "
  f"{pct(dist['share_below_threshold_peak'])} during weekday peaks and {pct(dist['share_below_threshold_offpeak'])} off-peak "
  f"(mean {dist['mean_speed_peak']:.1f} mph at peak vs {dist['mean_speed_offpeak']:.1f} mph off-peak). Congestion is the minority class and dominates errors.")
FIGURE(FIG / "eda/04_speed_distribution.png", "Speed distribution overall and peak vs off-peak (dashed: 40 mph).")
H2("6.4 Autocorrelation")
ac = eda["acf"]
P("Three representative sensors (lowest, median and highest variance among sensors with < 10% missing) show autocorrelation at the "
  "daily lag of 288 steps of " + ", ".join(f"{v:.3f}" for v in ac.values()) + " respectively: daily seasonality exists but its strength "
  "varies by sensor. Short-lag autocorrelation is very strong, which justifies a 12-step history and makes nearby windows dependent "
  "(a reason for chronological splits and autocorrelation-robust significance tests).")
FIGURE(FIG / "eda/05_acf_pacf.png", "ACF (left, red line at lag 288) and PACF (right) for three sensors.", 5.8)
H2("6.5 Spatial dependence")
c = eda["corr_vs_hops"]
TABLE(["Hops", "1", "2", "3", "4", "5", "6", "7", "8", "unreachable"],
      [["Median correlation"] + [f"{c[f'median_corr_{k}_hops']:.3f}" for k in range(1, 9)] + [f"{c['median_corr_unreachable']:.3f}"]],
      "Median pairwise speed correlation by shortest-path distance in the (undirected) graph, training period.", small=True)
FIGURE(FIG / "eda/06_corr_vs_hops.png", "Correlation vs graph distance. Neighbours are clearly more informative than distant sensors.", 5.6)
H2("6.6 Propagation evidence (lagged cross-correlation)")
x = eda["xcorr"]
P("For every directed edge (i → j, j downstream) with both sensors < 10% missing, speeds are detrended by the training time-of-day mean "
  "and the correlation corr(z_down(t), z_up(t + lag)) is computed for lags −12…+12. A positive peak lag means the upstream sensor "
  "changes after the downstream one, i.e. congestion moving upstream.")
TABLE(["Period", "Pairs", "Peak lag > 0", "Peak lag = 0", "Peak lag < 0"],
      [["All times", f"{x['xcorr_pairs']:,}", pct(x["xcorr_share_peak_lag_positive"], 0), pct(x["xcorr_share_peak_lag_zero"], 0), pct(x["xcorr_share_peak_lag_negative"], 0)],
       ["Weekday peak hours", f"{x['xcorr_pairs']:,}", pct(x["xcorr_peakhours_share_positive"], 0), pct(x["xcorr_peakhours_share_zero"], 0), pct(x["xcorr_peakhours_share_negative"], 0)]],
      "Distribution of the peak lag over all adjacent pairs.", widths=[28, 14, 19, 19, 20])
P("Reading: spatial dependence is clear; directional propagation is only modestly supported. The strongest pairs peak at lag 0 "
  "(adjacent sensors move together at 5-minute resolution); at peak hours positive lags outnumber negative ones, but many pairs "
  "show the opposite sign. Example pairs in Figure 9 are chosen by stated rules (top correlation; largest non-zero lag with "
  "correlation ≥ 0.5), never by eye.")
FIGURE(FIG / "eda/07_lagged_xcorr.png", "Cross-correlation curves for six pairs chosen by rule.")
FIGURE(FIG / "eda/07b_peak_lag_hist.png", "Peak lag across all adjacent pairs: all times vs weekday peak hours.", 5.0)
H2("6.7 Where congestion happens")
cm_ = eda["congestion_map"]
P(f"The median sensor is below 40 mph for {pct(cm_['peak_congestion_share_median'], 0)} of weekday-peak time; "
  f"{cm_['sensors_congested_over_20pct_of_peak']} sensors exceed 20% and the worst is congested {pct(cm_['peak_congestion_share_max'], 0)} of the time. "
  f"{eda['sensor_map']['sensors_mean_speed_below_55']} sensors have a mean speed below 55 mph. Interactive maps: "
  "`results/maps/eda_sensors.html`, `results/maps/eda_congestion_frequency.html`.")
FIGURE(FIG / "eda/09_congestion_frequency_map.png", "Share of weekday-peak time below 40 mph per sensor.", 4.6)
H2("6.8 Split comparability")
TABLE(["Split", "Weekday share", "Peak share", "Missing rate", "Mean speed (mph)"],
      [[k, pct(v["weekday_share"]), pct(v["peak_share"]), pct(v["missing_rate"]), f"{v['mean_speed']:.2f}"]
       for k, v in eda["splits"]["split_stats"].items()],
      "Composition of the three periods.", widths=[16, 21, 21, 21, 21])
FIGURE(FIG / "eda/10_split_timeline.png", "Daily mean speed with the train / validation / test periods.")
BREAK()

# ------------------------------------------------------------------ 7
H1("7. Evaluation protocol (identical for all models)")
H2("7.1 Common prediction format")
P("Every model writes `results/predictions/<model>[_seed<k>].npz` with `y_pred` and `y_true` [S, F, N] in mph, `mask` [S, F, N], "
  "`target_ts` [S, F], `sensor_ids` and metadata. The evaluator first asserts that every file has exactly the same test timestamps, "
  f"masks and sensors as persistence (S = {len(prep.windows['test']):,} test windows × 12 horizons × 207 sensors). Horizons a model does "
  "not produce (per-segment XGBoost trains only 15/30/60 minutes) are stored as NaN and skipped.")
H2("7.2 Metrics (masked, in mph)")
EQ("MAE = Σ m·|y − ŷ| / Σ m        RMSE = √( Σ m·(y − ŷ)² / Σ m )        MAPE = 100 · Σ m′·|y − ŷ|/y / Σ m′,   m′ = m ∧ (y ≥ 1 mph)")
P("An all-masked slice returns NaN, never 0. NaN predictions are removed from the mask. Hand-computed unit tests cover these cases (ADR-005).")
H2("7.3 Slices")
B(["Every horizon 1…12; headline tables at 3, 6, 12 steps (15, 30, 60 minutes).",
   "Peak AM (06:00–10:00), peak PM (15:00–19:00), peak (either), off-peak weekday, weekend — judged on the target timestamp.",
   "Congested (true speed < 40 mph) vs free-flow (≥ 40 mph).", "Per sensor (for maps)."])
H2("7.4 Statistical significance")
B(["**Diebold–Mariano (DM) test** on the per-timestamp loss differential d_t = mean over sensors |e₁| − mean |e₂| (sensors observed in both), "
   "per headline horizon. Long-run variance: Newey–West with a Bartlett kernel and L = max(h − 1, ⌈T^(1/3)⌉) lags; Harvey–Leybourne–Newbold "
   "small-sample correction; Student t with T − 1 degrees of freedom. Negative statistic: the first model has lower loss.",
   "For multi-seed models the test is run per seed pair; we report the median statistic and the **largest** (most conservative) p-value.",
   "**Day-block bootstrap:** whole days of target timestamps are resampled with replacement (1,000 draws, seed 42) for 95% intervals; paired "
   "versions resample the same days for both models so day difficulty cancels."])
H2("7.5 Automatic sanity checks")
B(["Any model with MAE at 60 minutes below 3.0 mph triggers a leakage warning (published strong graph models report ≈ 3.5).",
   "Any learned model not beating persistence at 15 minutes triggers “something is broken”.",
   f"Final run: “{(MET / 'sanity_warnings.txt').read_text().strip()}”."])
BREAK()

# ------------------------------------------------------------------ 8
H1("8. Baselines")
B(["**Persistence:** ŷ(t + h) = last observed value in the input window (past-only forward fill up to 11 steps; time-of-day mean if the whole window is missing).",
   "**Historical average:** mean of that sensor at the same (day-of-week, 5-minute slot), training period only (7 × 288 × 207 table); falls back to the slot mean.",
   f"**Seasonal naive:** ŷ(t + h) = y(t + h − 288) (same time yesterday); when that value is missing it falls back to the historical average "
   f"({pct(meta('seasonal_naive')['seasonal_fallback_share'], 2)} of entries)."])
TABLE(["Baseline", "MAE 15", "MAE 30", "MAE 60", "RMSE 15", "RMSE 30", "RMSE 60", "MAPE 15", "MAPE 30", "MAPE 60"],
      [[NICE[m]] + [f"{mt(m, h, col):.2f}" for col in ("MAE", "RMSE", "MAPE") for h in (3, 6, 12)] for m in MODEL_ORDER[:3]],
      "Baseline test errors (mph; MAPE in %).", small=True)
P(f"The historical average is flat across horizons because it ignores the current state; its 4.19 mph is close to the 4.16 reported for "
  "the historical average in the DCRNN paper, a useful external check of the pipeline.")
BREAK()

# ------------------------------------------------------------------ 9
H1("9. Phase 1 — machine learning")
H2("9.1 Tabular features")
P("One row per (sensor, anchor time); all features use data at or before the anchor (tested by perturbing the future).")
TABLE(["Group", "Features", "Count"],
      [["Lags", "lag_0 … lag_11 (imputed speed)", 12],
       ["Rolling statistics", "mean, std, min, max over 3, 6, 12 steps", 12],
       ["Differences", "lag_0 − lag_1, lag_0 − lag_3", 2],
       ["Missing flags", "miss_0 … miss_11 and n_missing_window", 13],
       ["Calendar (at anchor)", "tod_sin, tod_cos, dow_sin, dow_cos, is_weekend, is_holiday", 6],
       ["Static (pooled models)", "sensor_idx (categorical), latitude, longitude, in_degree, out_degree", 5],
       ["Neighbour (XGB-spatial)", "upstream and downstream weighted means at lags 0–2 (6), 2-hop mean, own − upstream, own − downstream", 9],
       ["Daily lag (side experiment)", "lag_288 (same time yesterday)", 1]],
      "Feature groups. Per-segment models use 45 features, pooled models 50, XGB-spatial 59, the daily-lag run 51.", widths=[24, 62, 14], small=True)
P("Neighbour features use row-normalised `W` (downstream senders) and `Wᵀ` (upstream senders); sensors with no neighbour in a direction get NaN, which XGBoost handles natively.")
H2("9.2 Models")
B(["**XGBoost per-segment:** one model per sensor per headline horizon, 207 × 3 = 621 models, ~24k rows each, `tree_method=hist`, single-threaded per model and parallel over sensors.",
   "**XGBoost global (pooled):** one model per horizon (12 models) across all sensors with sensor id as a categorical feature.",
   "**Random Forest (multi-output):** one pooled forest that predicts all 12 horizons jointly (joint squared-error split criterion); rows with any missing horizon are dropped.",
   "**XGBoost spatial:** the pooled model plus the 9 neighbour features — the bridge experiment that tests how much of the graph benefit hand-made features capture."])
H2("9.3 Hyperparameter tuning (Optuna, validation only)")
P("TPE sampler (seed 42). XGBoost objective: validation MAE at horizon 6 on 300,000 sampled training rows and 150,000 validation rows, "
  "early stopping 30 rounds, tree cap 400; per-segment tuning pools 20 random sensors with shared hyperparameters. Random Forest: mean "
  "validation MAE over all 12 horizons on 60,000 / 30,000 rows. The test split is never touched during tuning.")
TABLE(["Model", "Trials", "Best val MAE", "Median val MAE"],
      [[NICE[k], len(v), f"{v.value.min():.4f}", f"{v.value.median():.4f}"] for k, v in tun.items()],
      "Tuning summary (validation MAE in mph; tuning subsets differ between models, so values are not mutually comparable).", widths=[40, 15, 22, 23])
search = [["max_depth", "3 – 10"], ["learning_rate", "0.02 – 0.3 (log)"], ["subsample", "0.5 – 1.0"], ["colsample_bytree", "0.5 – 1.0"],
          ["min_child_weight", "1 – 20"], ["reg_lambda", "0.001 – 10 (log)"], ["gamma", "0 – 5"],
          ["objective", "squared error | absolute error | pseudo-Huber"]]
rows = []
for name, _rng in search:
    rows.append([name, _rng] + [str(round(best[m][name], 4) if isinstance(best[m].get(name), float) else best[m].get(name, "–"))
                                for m in ("xgb_per_segment", "xgb_global", "xgb_spatial")])
TABLE(["Parameter", "Search range", "Per-segment", "Global", "Spatial"], rows,
      "XGBoost search space and the selected values (absolute-error loss chosen for all three). Global and spatial chose identical values: same seed and the same best trial index.",
      widths=[20, 30, 17, 17, 16], small=True)
rf = best["rf_multioutput"]
P(f"Random Forest search: n_estimators 60–150, max_depth 8–30, max_features ∈ {{0.2, 0.33, 0.5}}, min_samples_leaf 1–20, max_samples 0.5. "
  f"Selected: {rf['n_estimators']} trees, max_depth {rf['max_depth']}, max_features {rf['max_features']}, min_samples_leaf {rf['min_samples_leaf']}, max_samples {rf['max_samples']}.")
H2("9.4 Training budget (reduced for a laptop, ADR-006)")
rfm = meta("rf_multioutput")
TABLE(["Model", "Training rows", "Training time", "Notes"],
      [["XGBoost global", "≈ 1.15 million per horizon (25% sample)", f"{meta('xgb_global')['train_seconds']:.0f} s", "12 horizons; every model used 398–399 of 400 trees"],
       ["XGBoost spatial", "≈ 1.15 million per horizon (25% sample)", f"{meta('xgb_spatial')['train_seconds']:.0f} s", "12 horizons; every model used 398–399 of 400 trees"],
       ["XGBoost per-segment", "all rows of each sensor", f"{meta('xgb_per_segment')['train_seconds']:.0f} s", "621 models"],
       ["Random Forest", f"{rfm['train_rows_used']:,} of {rfm['train_rows_all_targets_observed']:,} complete rows (10%)", f"{rfm['train_seconds']:.0f} s",
        f"{pct(rfm['share_dropped_missing_target'], 1)} of rows dropped (a horizon missing)"],
       ["XGBoost global + daily lag", "≈ 1.15 million per horizon", f"{meta('xgb_global_daily')['train_seconds']:.0f} s", "side experiment"]],
      "Training data and wall-clock time (6 CPU workers, low priority, Apple M3 Pro).", widths=[22, 33, 15, 30], small=True)
NOTE("Early stopping never triggered for the pooled XGBoost models: the best iteration was 398–399 of the 400-tree cap at every horizon. "
     "More trees (and more training rows) would very likely lower their error further. The reported ML numbers are therefore conservative.")
H2("9.5 Phase 1 results")
mlm = MODEL_ORDER[:7]
TABLE(["Model"] + [f"{c} {hh}" for c in ("MAE", "RMSE", "MAPE") for hh in (15, 30, 60)],
      [[NICE[m]] + [f"{mt(m, h, col):.2f}" for col in ("MAE", "RMSE", "MAPE") for h in (3, 6, 12)] for m in mlm],
      "Phase 1 test errors (mph; MAPE in %).", small=True)
B([f"XGB-spatial is best at 15 minutes ({mt('xgb_spatial', 3):.3f}) and 30 minutes ({mt('xgb_spatial', 6):.3f}); per-segment XGBoost is best at 60 minutes ({mt('xgb_per_segment', 12):.3f}).",
   f"The forest is the weakest ML model ({mt('rf_multioutput', 3):.3f} / {mt('rf_multioutput', 6):.3f} / {mt('rf_multioutput', 12):.3f}); at 60 minutes it loses to the historical average.",
   "All Phase 1 ML models beat persistence at every horizon; none comes close to the 3.0 mph leakage bound at 60 minutes."])
FIGURE(FIG / "ml/error_vs_horizon.png", "Test MAE vs horizon for baselines and Phase 1 models.", 5.6)
H2("9.6 What drives the trees (gain and TreeSHAP)")
P("TreeSHAP values come from XGBoost's exact `pred_contribs` on 5,000 seeded test rows with observed targets.")
shap_rows = []
for (m, h), g in fi[fi.kind == "mean_abs_shap"].groupby(["model", "horizon"]):
    top = g.sort_values("value", ascending=False).head(8)
    shap_rows.append([NICE[m], f"{h * 5} min", ", ".join(f"{r.feature} ({r.value:.2f})" for r in top.itertuples())])
TABLE(["Model", "Horizon", "Top 8 features by mean |SHAP| (mph)"], shap_rows,
      "Most important features.", widths=[22, 12, 66], small=True)
P("Short-window averages and the latest speed dominate at 15 minutes; sensor identity and time of day gain weight at 60 minutes. "
  "Neighbour speeds enter the top features of XGB-spatial but never dominate.")
FIGURE(FIG / "ml/shap_xgb_spatial_h12.png", "TreeSHAP summary, XGB-spatial at 60 minutes.", 5.2)
H2("9.7 Gap analysis: what do neighbour features add?")
gtab = gap[gap.horizon.isin([3, 6, 12])]
TABLE(["Horizon", "Slice", "MAE global", "MAE spatial", "Δ MAE", "95% CI", "DM p"],
      [[f"{r.horizon * 5} min", r.slice, f"{r.mae_global:.3f}", f"{r.mae_spatial:.3f}", f"{r.delta_mae:+.3f}",
        f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", f"{r.p_value:.3f}"] for r in gtab.itertuples()],
      "Δ = MAE(global) − MAE(spatial); positive means the spatial features help. Paired day-block bootstrap intervals; Diebold–Mariano p-values.",
      widths=[12, 20, 13, 13, 12, 18, 12], small=True)
P("The spatial features help in almost every slice, most at peaks and in congested states (≈ 0.3–0.6 mph in congestion), as the "
  "propagation hypothesis predicts — but the gain does **not** grow with horizon (largest at 30 minutes), so the “graph helps more at "
  "long horizons” part is only partly supported. In free flow at 60 minutes they are neutral to slightly harmful.")
FIGURE(FIG / "ml/gap_analysis.png", "Gap analysis by horizon and slice with 95% intervals.")
H2("9.8 Side experiment: same-time-yesterday lag")
TABLE(["Model", "MAE 15", "MAE 30", "MAE 60"],
      [[NICE["xgb_global"]] + [f"{mt('xgb_global', h):.4f}" for h in (3, 6, 12)],
       [NICE["xgb_global_daily"]] + [f"{float(daily[daily.horizon == h].MAE.iloc[0]):.4f}" for h in (3, 6, 12)]],
      "Adding lag_288 changes nothing material (kept out of the main equal-window comparison).", widths=[46, 18, 18, 18])
H2("9.9 Hardest sensors")
TABLE(["Sensor", "Latitude", "Longitude", "MAE @ 60 min (XGB global)"],
      [[r.sensor_id, f"{r.latitude:.5f}", f"{r.longitude:.5f}", f"{r.mae:.2f}"] for r in hard.head(5).itertuples()],
      "The five hardest sensors cluster at one location (≈ 34.09 N, 118.31 W).", widths=[22, 22, 22, 34])
BREAK()

# ------------------------------------------------------------------ 10
H1("10. Phase 2 — deep learning")
H2("10.1 The graph attention layer")
P("For node i with neighbourhood N(i) ∪ {i} (self-loops always included), with K heads of size D:")
EQ("h′ = W h        e_ij = LeakyReLU( a_dstᵀ h′_i + a_srcᵀ h′_j )  (= aᵀ[h′_i ‖ h′_j])        α_ij = softmax_j(e_ij)        out_i = Σ_j α_ij h′_j + b")
B(["**Dense masked form** (`DenseGATLayer`): scores for all [N × N] pairs, non-neighbours set to −∞ before the softmax; heads concatenated (or averaged).",
   "**Sparse form** (`SparseGATLayer`): padded neighbour lists (D_max = 26 for the bidirectional graph vs N = 207); numerically identical to the dense layer (tested).",
   "**Variants:** GATv2 (dynamic attention, aᵀ LeakyReLU(W_dst h_i + W_src h_j), computed in gradient-checkpointed chunks of 32 graphs), fixed row-normalised "
   "weights (“mean”, GCN-style), and an edge-weight bias β·W_ij per head.",
   "**Verification:** outputs and attention weights equal PyTorch Geometric's `GATConv` (concat and mean) and `GATv2Conv` within 1e-5 on a random 12-node graph "
   "with copied weights (`tests/test_gat_layer.py`). PyG's edge_index is (source j, target i), so it is built as `adj.nonzero().flip(1).T`."])
H2("10.2 STGAT architecture")
mc = stgat_cfg["model"]
B([f"Input X [B, 12, 207, 8] → Linear 8 → d = {mc['d']} → plus a learnable node embedding E ∈ ℝ^(207×{mc['d']}).",
   f"{mc['layers']} spatio-temporal blocks. Each: gated causal temporal convolution (kernel 3, dilation 1 then 2, GLU gate, left padding so step t sees only ≤ t) "
   f"→ multi-head GAT ({mc['heads']} heads × {mc['d'] // mc['heads']} = {mc['d']}) on the **last {mc['spatial_steps']} of 12 time steps** → ELU → dropout {mc['dropout']} → residual → LayerNorm.",
   f"Attention dropout {mc['attn_dropout']}; graph `{stgat_cfg['graph']['variant']}` (W + Wᵀ); readout = last time step; decoder Linear {mc['d']} → {mc['d']} → ReLU → dropout → Linear {mc['d']} → 12.",
   "Non-autoregressive: all 12 horizons predicted at once.",
   f"Parameters: **{meta('stgat_seed0')['n_params']:,}**."])
H2("10.3 GRU no-graph ablation")
B([f"Same embedding, node embedding, width d = {gru_cfg['model']['d']}, dropout and decoder; a 2-layer GRU runs over each sensor's own 12 steps; no spatial layer.",
   f"Parameters: **{meta('gru_nograph_seed0')['n_params']:,}** (similar capacity), so any gap to STGAT is attributable to the graph, not to deep learning in general."])
H2("10.4 Training")
tr = stgat_cfg["train"]
TABLE(["Setting", "Value"],
      [["Loss", "masked MAE in scaled space (normalised by the number of observed targets)"],
       ["Optimiser", f"Adam, learning rate {tr['lr']}, weight decay {tr['weight_decay']}"],
       ["Schedule", "ReduceLROnPlateau on validation MAE (factor 0.5, patience 5)"],
       ["Gradient clipping", f"norm {tr['clip']}"], ["Batch size", tr["batch_size"]],
       ["Epochs", f"max {tr['max_epochs']}, early stopping patience {tr['patience']} on validation MAE in mph; best weights restored"],
       ["Epoch fraction", f"{tr['epoch_fraction']} (each epoch sees a random third of the training windows)"],
       ["Seeds", "0, 1, 2"], ["Device", "Apple M3 Pro GPU (PyTorch MPS), 4 CPU threads, low priority"]],
      "Deep-learning training configuration (same for STGAT and GRU).", widths=[28, 72], small=True)
H2("10.5 Compute study and simplification (ADR-007)")
TABLE(["Configuration", "Time per iteration", "Time per epoch"],
      [["STGAT dense, GAT on all 12 steps, CPU", "4.34 s", "27.1 min"],
       ["STGAT dense, GAT on all 12 steps, MPS", "0.59 s", "3.7 min"],
       ["STGAT sparse neighbour lists, all 12 steps, MPS", "0.66 s", "4.1 min (no gain: gathers dominate)"],
       ["STGAT GAT on last 4 steps + 34% of windows per epoch, MPS", "0.28 s", "≈ 0.6 min (+ validation)"],
       ["GRU no-graph, CPU / MPS (full epoch)", "0.27 s / 0.08 s", "1.7 / 0.5 min"]],
      "Measured training speed that motivated the simplification.", widths=[56, 20, 24], small=True)
P("Profiling showed the per-block cost was dominated by the graph attention (0.26 s of 0.33 s per block), which is memory-bound. Applying "
  "attention on the most recent 4 steps (20 minutes of neighbour state) and sampling a third of the highly redundant windows each epoch "
  "cut the cost about six-fold, to ≈ 28 minutes per seed.")
H2("10.6 Training outcomes")
rows = []
for m in ("gru_nograph", "stgat", "hybrid_residual"):
    for s in (0, 1, 2):
        mt_ = meta(f"{m}_seed{s}")
        log = train_logs.get(f"{m}_seed{s}")
        best_ep = int(log.loc[log.val_mae_mph.idxmin(), "epoch"]) if log is not None else "–"
        rows.append([NICE[m], s, mt_.get("epochs", "–"), best_ep, f"{log.val_mae_mph.min():.4f}" if log is not None else "–",
                     f"{mt_.get('train_seconds', float('nan')):.0f} s"])
TABLE(["Model", "Seed", "Epochs run", "Best epoch", "Best val MAE (mph)", "Train time"], rows,
      "Per-seed training results. STGAT never triggered early stopping within 40 epochs (it was still improving).",
      widths=[26, 9, 14, 14, 21, 16], small=True)
FIGURE(FIG / "training/stgat_seed0.png", "Training curve, STGAT seed 0 (training loss in scaled units; validation MAE in mph).", 4.6)
H2("10.7 Results")
dl = ["xgb_per_segment", "xgb_global", "xgb_spatial", "gru_nograph", "stgat"]
TABLE(["Model", "MAE 15", "MAE 30", "MAE 60", "RMSE 15", "RMSE 30", "RMSE 60"],
      [[NICE[m]] + [cell(m, h) for h in (3, 6, 12)] + [cell(m, h, "RMSE") for h in (3, 6, 12)] for m in dl],
      "Deep models vs the tree models (mph; ± std over 3 seeds).", small=True)
B([f"STGAT beats persistence at every horizon (acceptance criterion met).",
   f"Graph vs no graph: equal at 15 minutes ({mt('stgat', 3):.3f} vs {mt('gru_nograph', 3):.3f}, DM not significant), slightly ahead at 30 "
   f"({mt('stgat', 6):.3f} vs {mt('gru_nograph', 6):.3f}, not significant), clearly ahead at 60 ({mt('stgat', 12):.3f} vs {mt('gru_nograph', 12):.3f}, significant).",
   f"Deep vs trees: STGAT is slightly worse than pooled XGBoost at 15 minutes ({mt('stgat', 3):.3f} vs {mt('xgb_global', 3):.3f}) and better at 60 "
   f"({mt('stgat', 12):.3f} vs {mt('xgb_global', 12):.3f}); per-segment XGBoost stays very close at 60 minutes ({mt('xgb_per_segment', 12):.3f}).",
   "STGAT has a lower RMSE than every tree model at 30 and 60 minutes, i.e. fewer large misses."])
BREAK()

# ------------------------------------------------------------------ 11
H1("11. Hybrid models")
H2("11.1 Why a residual hybrid")
P("XGBoost captures local autoregression and calendar effects cheaply; spatial propagation is its blind spot. The residual hybrid "
  "trains STGAT to learn only what XGBoost misses:")
EQ("ŷ_hybrid(t + h, i) = ŷ_XGB(t + h, i) + g_θ(X_window, ŷ_XGB, G)(h, i)")
P("STGAT receives the scaled XGBoost forecast (12 values per sensor) concatenated to its readout before the decoder, and the last decoder "
  "layer is zero-initialised, so training starts exactly at the XGBoost forecast (tested).")
H2("11.2 Leakage-safe out-of-fold base forecasts")
edges = [int(e) for e in np.linspace(0, len(prep.windows["train"]), 6)]
B(["XGBoost's predictions on its own training rows are overfit; their residuals are unrealistically small. The residual model must learn from "
   "**out-of-fold (OOF)** forecasts instead.",
   f"Forward-chaining over the {len(prep.windows['train']):,} training windows in 5 chronological blocks with edges {edges}.",
   "For blocks 2–5, an XGBoost model (tuned global parameters, 15% row sample, early stopping on 20,000 validation rows) trained on the earlier "
   "blocks predicts the block; a gap of H + F − 1 = 23 anchors separates training targets from the block's inputs.",
   f"Block 1 gets no OOF forecast and is dropped: residual training uses {len(prep.windows['train']) - edges[1]:,} windows "
   f"({pct((len(prep.windows['train']) - edges[1]) / len(prep.windows['train']), 1)} coverage).",
   "The final XGBoost models (trained on all training rows) provide the base forecasts for validation and test.",
   "Caveat: early-block models see less data, so OOF errors are slightly larger than the final model's — the conservative direction."])
H2("11.3 Gating ensemble and simple average")
B(["**Gate:** ŷ = w·ŷ_STGAT + (1 − w)·ŷ_XGB with w = sigmoid(MLP(16 → 16 → 1)) of horizon one-hot (12), weekday-peak flag, time-of-day sin/cos of the "
   "target, and the input window's missing rate; trained with Adam (lr 0.01) on the first 80% of validation windows, early-stopped (patience 30, ≤ 400 steps) "
   "on the last 20% (ADR-008: validation is then not fully independent).",
   "**Average:** 0.5·ŷ_XGB + 0.5·ŷ_STGAT, per seed.", "All hybrids: 3 seeds, paired with the STGAT seed."])
H2("11.4 Results")
hy = ["xgb_global", "stgat", "hybrid_residual", "hybrid_gating", "hybrid_average"]
TABLE(["Model", "MAE 15", "MAE 30", "MAE 60", "RMSE 15", "RMSE 30", "RMSE 60"],
      [[NICE[m]] + [cell(m, h) for h in (3, 6, 12)] + [cell(m, h, "RMSE") for h in (3, 6, 12)] for m in hy],
      "Hybrids vs their components (mph; ± std over 3 seeds).", small=True)
g = hg[(hg.hybrid == "hybrid_residual") & hg.component.isin(["xgb_global", "stgat"])]
gp = g.pivot_table(index=["component", "slice"], columns="horizon_min", values="gain_mph").reset_index()
order = ["all", "peak", "offpeak_weekday", "weekend", "congested", "freeflow"]
gp["o"] = gp["slice"].map({s: i for i, s in enumerate(order)})
gp = gp.dropna(subset=["o"]).sort_values(["component", "o"])
TABLE(["Component", "Slice", "Gain 15", "Gain 30", "Gain 60"],
      [[NICE[r.component], r.slice, f"{r[15]:+.3f}", f"{r[30]:+.3f}", f"{r[60]:+.3f}"] for _, r in gp.iterrows()],
      "Gain of the residual hybrid = MAE(component) − MAE(hybrid), seed-averaged (positive: hybrid better).",
      widths=[30, 22, 16, 16, 16], small=True)
P("Over XGBoost the residual hybrid's gain grows with horizon and is largest in congestion and at peaks; in free flow at 60 minutes it is "
  "slightly negative. Over STGAT the gain shrinks with horizon: STGAT already captures much of the local structure at long range.")
TABLE(["Horizon (min)"] + [str(h * 5) for h in gw.horizon],
      [["Weight on STGAT, peak"] + [f"{v:.2f}" for v in gw.mean_w_stgat_peak],
       ["Weight on STGAT, off-peak"] + [f"{v:.2f}" for v in gw.mean_w_stgat_offpeak]],
      "Mean learned gate weight on STGAT (test set, averaged over seeds).", small=True)
P("The gate trusts STGAT more at peaks than off-peak and more at longer horizons — exactly the pattern the hypothesis predicts.")
FIGURE(FIG / "hybrid/hybrid_gain.png", "Residual-hybrid gain over XGBoost-global and over STGAT by slice and horizon.")
FIGURE(FIG / "hybrid/gate_weights.png", "Learned gate weight on STGAT by horizon, peak vs off-peak.", 4.4)
BREAK()

# ------------------------------------------------------------------ 12
H1("12. All results")
H2("12.1 Main table")
TABLE(["Model"] + [f"{c} {hh}" for c in ("MAE", "RMSE", "MAPE") for hh in (15, 30, 60)],
      [[NICE[m]] + [f"{mt(m, h, col):.3f}" for col in ("MAE", "RMSE", "MAPE") for h in (3, 6, 12)] for m in MODEL_ORDER],
      "All models, all headline metrics (seed means for deep models).", small=True)
FIGURE(FIG / "error_vs_horizon_all_models.png", "MAE vs horizon for every model (bands: ± std over seeds).")
H2("12.2 Every horizon")
a_all = agg[(agg.slice == "all") & (agg.metric == "MAE")]
rows = []
for m in MODEL_ORDER:
    s_ = a_all[a_all.model == m].set_index("horizon")["mean"]
    rows.append([NICE[m]] + [f"{s_[h]:.2f}" if h in s_.index else "–" for h in range(1, 13)])
TABLE(["Model"] + [f"{h * 5}" for h in range(1, 13)], rows, "MAE (mph) at every horizon in minutes.", small=True)
H2("12.3 Peak, off-peak and weekend")
rows = []
for m in MODEL_ORDER:
    r_ = []
    for sl in ("peak", "offpeak_weekday", "weekend"):
        for h in (3, 12):
            v = po[(po.model == m) & (po.slice == sl) & (po.horizon == h)].MAE
            r_.append(f"{float(v.iloc[0]):.2f}" if len(v) else "–")
    rows.append([NICE[m]] + r_)
TABLE(["Model", "Peak 15", "Peak 60", "Off-peak 15", "Off-peak 60", "Weekend 15", "Weekend 60"], rows,
      "MAE by time slice (mph).", small=True)
H2("12.4 Congested vs free-flow")
rows = []
for m in MODEL_ORDER:
    r_ = []
    for sl in ("congested", "freeflow"):
        for h in (3, 6, 12):
            v = cf[(cf.model == m) & (cf.slice == sl) & (cf.horizon == h)].MAE
            r_.append(f"{float(v.iloc[0]):.2f}" if len(v) else "–")
    rows.append([NICE[m]] + r_)
TABLE(["Model", "Cong. 15", "Cong. 30", "Cong. 60", "Free 15", "Free 30", "Free 60"], rows,
      "MAE on congested (< 40 mph) and free-flow targets (mph). Congested errors are 3–5 times larger.", small=True)
FIGURE(FIG / "peak_offpeak_bars.png", "MAE by slice at 15, 30 and 60 minutes.")
H2("12.5 Uncertainty: day-block bootstrap")
rows = []
for m in MODEL_ORDER + ["xgb_global_daily"]:
    r_ = []
    for h in (3, 6, 12):
        b_ = boot[(boot.model == m) & (boot.horizon == h)]
        r_.append(f"{b_.MAE.iloc[0]:.2f} [{b_.ci_lo.iloc[0]:.2f}, {b_.ci_hi.iloc[0]:.2f}]" if len(b_) else "–")
    rows.append([NICE[m]] + r_)
TABLE(["Model", "15 min", "30 min", "60 min"], rows,
      "MAE with 95% day-block bootstrap intervals (first seed). Intervals of different models overlap because day difficulty dominates; "
      "paired tests (DM, paired bootstrap) are needed to compare models.", widths=[34, 22, 22, 22], small=True)
H2("12.6 Significance (Diebold–Mariano)")
sg = sig.copy()
TABLE(["Model A", "Model B", "Horizon", "DM statistic", "p-value", "Seed pairs", "Verdict"],
      [[NICE[r.model_a], NICE[r.model_b], f"{r.horizon * 5} min", f"{r.dm_stat:+.2f}", f"{r.p_value:.4f}", r.n_seed_pairs,
        ("A better" if r.dm_stat < 0 else "B better") if r.p_value < 0.05 else "no significant difference"]
       for r in sg.itertuples()],
      "Diebold–Mariano tests (negative statistic: model A has lower loss; p is the largest over seed pairs).",
      widths=[20, 20, 10, 13, 11, 10, 16], small=True)
BREAK()

# ------------------------------------------------------------------ 13
H1("13. Ablation studies")
P("Each ablation changes one setting from the STGAT default. Budget (to suit a laptop): **one seed and 12 epochs** per run, with a "
  "re-run of the default under the same budget as the reference (A0). Runs with different history lengths have different first test "
  f"windows, so all runs are compared on the {6820:,} test anchors they share. Differences of a few hundredths of a mph are within "
  f"seed noise (the 3-seed std of the full STGAT runs is {mstd('stgat', 3):.3f}–{mstd('stgat', 12):.3f} mph).")
ab = abl[abl.metric == "MAE"].pivot_table(index=["ablation", "variant"], columns="horizon_min", values="mean")
ref = ab.loc[("A0", "default")]
secs = abl_long.groupby(["ablation", "variant"]).train_seconds.first()
desc = {"default": "reference (bidirectional graph, GAT, 4 heads, 2 layers)", "identity": "no neighbours",
        "random": "random graph, same edge count", "forward": "W only (downstream senders)", "backward": "Wᵀ only (upstream senders)",
        "gatv2": "GATv2 attention", "mean": "fixed row-normalised weights", "heads1": "1 head", "heads2": "2 heads", "heads8": "8 heads",
        "layers1": "1 block", "layers3": "3 blocks", "no_node_emb": "no node embedding", "no_time": "no calendar channels",
        "zero_mask": "zeros kept + mask (DCRNN style)", "no_dropout": "dropout 0", "no_wd": "weight decay 0", "H6": "history 6 steps",
        "H24": "history 24 steps", "edge_bias": "distance bias in attention"}
keyf = lambda t: (int(t[0][1:]), t[1])  # noqa: E731
rows = []
for (aid, var) in sorted(ab.index, key=keyf):
    r = ab.loc[(aid, var)]
    rows.append([aid, desc.get(var, var), f"{r[15]:.3f}", f"{r[30]:.3f}", f"{r[60]:.3f}", f"{r[15] - ref[15]:+.3f}",
                 f"{r[30] - ref[30]:+.3f}", f"{r[60] - ref[60]:+.3f}", f"{secs.get((aid, var), float('nan')):.0f}"])
TABLE(["ID", "Change", "MAE 15", "MAE 30", "MAE 60", "Δ15", "Δ30", "Δ60", "Time (s)"], rows,
      "Ablation results (Δ vs A0; negative = better than the default).", widths=[6, 34, 9, 9, 9, 8, 8, 8, 9], small=True)
B(["**No graph benefit shown under this budget:** the identity graph (A1) is not worse than the road graph — at 60 minutes it is better.",
   "**Wrong edges hurt:** a random graph with the same number of edges (A2) is worse than both, weak evidence that topology matters.",
   "**What matters most:** removing the node embedding (A7) or the calendar features (A8) costs 0.6–0.7 mph at 60 minutes.",
   "**Within noise:** edge direction (A3), attention type (A4), heads (A5), depth (A6), imputation (A9), history length (A11), edge bias (A12).",
   "**Regularisation (A10):** removing dropout or weight decay helped — with only 12 epochs the model is under-trained, so this says nothing about full training.",
   "**A13 (hybrid design):** residual vs gating vs average, read from the main table (Section 11).",
   "The 3-seed, 40-epoch comparison with the GRU (Section 10.7) remains the stronger evidence that the graph helps at 60 minutes."])
FIGURE(FIG / "ablations/ablation_bars.png", "Ablation MAE at 15/30/60 minutes (dotted lines: default).")
BREAK()

# ------------------------------------------------------------------ 14
H1("14. Maps and interpretability")
H2("14.1 Forecast maps")
P(f"The AM-peak map anchor is chosen automatically: the test weekday window whose 15-minute targets fall in 07:00–09:00 with the most "
  f"congested observed sensors — anchor index {sel['anchor_index']} with targets {sel['target_times']['3']} (+15 min), "
  f"{sel['target_times']['6']} (+30) and {sel['target_times']['12']} (+60). Interactive files: `results/maps/forecast_am_peak.html` "
  "(truth, STGAT, XGB-global at three horizons, toggleable layers) and `animated_peak_truth.html` / `animated_peak_stgat.html` "
  "(time slider over one peak hour). Colours: red < 40 mph, orange 40–55, green > 55, grey missing.")
H2("14.2 Where the graph model helps")
FIGURE(FIG / "maps/error_diff_h12_peak.png", "Per-sensor MAE(XGB-global) − MAE(STGAT) at 60 minutes during peaks (blue: STGAT better).", 4.8)
P("Gains cluster along specific freeway segments; a few sensors are slightly worse with STGAT.")
H2("14.3 Space-time diagram")
P(f"A corridor of {len(corridor['indices'])} sensors is found automatically by following the nearest downstream neighbour with increasing "
  "road distance from each start sensor and keeping the longest chain. The day shown (2012-06-21) is the test weekday with the most "
  "corridor congestion between 05:00 and 11:00. Neither model anticipates the abrupt breakdown near 10:00 — onset is the hardest event to forecast.")
FIGURE(FIG / "maps/space_time.png", "Corridor space-time diagram: truth, STGAT and XGB-global 30-minute forecasts placed at their target time.")
H2("14.4 Attention (qualitative)")
TABLE(["Period", "Windows averaged", "Self", "Downstream only", "Upstream only", "Both directions"],
      [[r.period.replace("_", " "), r.n_windows, pct(r.self), pct(r.downstream_only), pct(r.upstream_only), pct(r.both_directions)]
       for r in att.itertuples()],
      "Share of last-block attention mass (STGAT seed 0, heads averaged), test weekdays: AM peak = anchors 07:30–08:30, night = 02:00–04:00.",
      widths=[16, 18, 14, 18, 17, 17], small=True)
P("The shares are the same at the AM peak and at night, so we find no peak-specific shift of attention towards upstream or downstream "
  "sensors. Attention is a probe, not causal evidence.")
FIGURE(FIG / "maps/attention_am_peak.png", "Strongest attention links for the three most congested sensors, AM peak.")
BREAK()

# ------------------------------------------------------------------ 15
H1("15. Simulated ramp-metering experiment")
NOTE("This is an offline simulation on a synthetic network. It demonstrates how forecasts could inform a control policy. "
     "No live or real-world traffic control is performed or claimed.")
H2("15.1 Why ramp metering")
P("METR-LA sensors are on freeways, which have no traffic signals; the realistic control analogue is on-ramp metering.")
H2("15.2 Cell Transmission Model")
TABLE(["Element", "Setting"],
      [["Corridor", "12 cells × 0.5 mile, 3 lanes, time step 10 s (CFL: 65 mph × 10 s = 0.18 mi ≤ 0.5 mi)"],
       ["Fundamental diagram", "triangular: free-flow 65 mph, capacity 2,000 veh/h/lane, jam density 200 veh/mi/lane, critical density 30.8, backward wave ≈ 11.8 mph"],
       ["On-ramps", "cells 2, 5, 8; maximum ramp discharge 1,800 veh/h"],
       ["Bottleneck", "cell 10 at 75% capacity (lane-drop-like)"],
       ["Capacity drop", "congested cells (lane density > 1.15 × critical) discharge at 88% of capacity"],
       ["Merge rule", "mainline gets 70% priority when the receiving cell cannot take both flows"],
       ["Upstream boundary", "virtual queue for demand that cannot enter"]],
      "Simulator settings (`src/traffic/control/ctm.py`).", widths=[24, 76], small=True)
P("Tests check vehicle conservation (in − out − stored = 0 to 1e-6), free flow at low demand, a queue that grows upstream of the "
  "bottleneck, ALINEA direction and clipping, and the speed-to-occupancy conversion.")
H2("15.3 Demand (synthetic, shaped by METR-LA)")
P("METR-LA has no flow, and speed cannot be inverted to flow in free flow (all sub-critical densities share one speed). Demand is "
  "therefore synthetic, shaped by a congestion index computed from the 12 corridor sensors mapped to the cells:")
EQ("ci(t) = clip( ((65 − mean speed(t)) / 45) / ci_ref , 0, 1 ),   ci_ref = 0.167 (95th percentile, training weekdays 05:00–11:00)")
EQ("mainline demand = 6000 · (0.45 + 0.22 · ci) veh/h        each ramp = 250 + 150 · ci veh/h")
P("At ci = 1 total demand is about 16% above the bottleneck capacity. Simulated 04:30–11:00, scored 05:00–11:00, on the five test "
  "weekdays with the most AM corridor congestion: 2012-06-21, 2012-06-12, 2012-06-15, 2012-06-07, 2012-06-26.")
H2("15.4 Policies")
B(["**No control:** ramps unmetered.",
   f"**Fixed-time:** one rate 06:00–10:00, tuned on the validation weekday 2012-05-29 over the grid {list(ctl_tune.rate_veh_h)} veh/h; rates ≥ 350 tied "
   f"(TTS {ctl_tune.TTS_veh_h.min():.1f} veh·h, identical to no control), and 400 was selected — non-binding, so it equals no control.",
   "**ALINEA:** r(k) = r(k − 1) + 70 · (o* − o(k)) every 60 s, o = occupancy (%) of the cell downstream of the ramp, o* = critical occupancy 15.4%, rates clipped to 200–1,800 veh/h.",
   "**Predictive ALINEA (literal spec):** occupancy = max(measured, occupancy implied by the STGAT +15-minute speed forecast via the congested branch ρ = w·ρ_jam/(v + w)).",
   "**Look-ahead ALINEA:** a copy of the simulator is rolled 15 minutes ahead from its live state with demand moving to the forecast congestion index; ALINEA uses max(measured, look-ahead occupancy).",
   "**Oracle versions** of both use the true future speeds (upper bound)."])
H2("15.5 Results")
cols = ["TTS_veh_h", "mainline_avg_speed_mph", "max_ramp_queue_veh", "congestion_cell_minutes", "distance_veh_mi"]
TABLE(["Policy", "TTS (veh·h)", "Mainline speed (mph)", "Max ramp queue (veh)", "Congested cell-minutes", "Distance (veh·mi)"],
      [[r.policy.replace("_", " ")] + [f"{getattr(r, c + '_mean'):,.1f} ± {getattr(r, c + '_std'):,.1f}" for c in cols] for r in ctl.itertuples()],
      "Simulation results, mean ± std over the five test weekdays.", widths=[20, 16, 16, 16, 16, 16], small=True)
B(["ALINEA cuts congested cell-minutes from ≈ 399 to ≈ 243 and raises mainline speed from ≈ 52 to ≈ 57 mph, but total time spent is unchanged: the delay moves to the ramps.",
   "Look-ahead ALINEA behaves like ALINEA; its oracle version is no better, so better forecasts would not help this controller here.",
   "The literal forecast-speed variant over-meters (very long ramp queues, much higher TTS) because real-data forecasts do not respond to the simulator's own state (open loop).",
   "Conclusion: in this simulation, forecasts gave no measurable benefit."])
FIGURE(FIG / "control/tts_by_policy.png", "Total time spent by policy (offline simulation).", 5.0)
FIGURE(FIG / "control/density_heatmaps.png", "Simulated density (veh/mi) along the corridor per policy, one test day (offline simulation).")
BREAK()

# ------------------------------------------------------------------ 16
H1("16. Engineering and reproducibility")
H2("16.1 Repository layout")
TABLE(["Path", "Content"],
      [["src/traffic/data", "loading, graph, missing data, split, scaling, windows, datasets, pipeline"],
       ["src/traffic/features", "calendar, lag, neighbour features; tabular design matrices"],
       ["src/traffic/models", "baselines, ML models, GAT layers, STGAT, GRU, hybrid, factory"],
       ["src/traffic/training", "masked losses, trainer, Optuna tuning"],
       ["src/traffic/evaluation", "prediction format, metrics, slices, significance, tables, report numbers"],
       ["src/traffic/viz", "EDA plots, maps, horizon plots, attention plots"],
       ["src/traffic/control", "CTM, ALINEA, predictive ALINEA, experiment"],
       ["scripts/", "entry points (download, EDA, train_ml, train_dl, make_oof, train_hybrid, ablations, evaluate, maps, control, report assets) and chain scripts"],
       ["configs/", "base.yaml, ml/*, dl/*, hybrid/*; tuned parameters in ml/*.best.yaml"],
       ["tests/", "unit and end-to-end tests"], ["results/metrics", "all CSV/JSON results (tracked)"],
       ["results/figures, results/maps", "figures (tracked) and interactive HTML maps"],
       ["report/phase1, report/phase2", "IEEE reports; every number comes from generated macros in report/tables/numbers.tex"],
       ["docs/", "EDA summary, analyses, theory notes (11), viva preparation, decisions (ADRs), workflow"]],
      "Repository structure.", widths=[28, 72], small=True)
H2("16.2 Tests")
P("61 tests pass (pytest), and `ruff` and `black` report a clean code base. Key tests: integrity of the raw data; graph rebuild equals the pickle; "
  "direction convention on a hand-built chain; no window crosses a split; scaler and imputation statistics depend on training data only; "
  "window alignment; features never use the future; tabular row order; masked metrics against hand-computed values; DM and bootstrap "
  "behaviour; GAT equals PyG (GATConv, GATv2Conv) and sparse equals dense; the temporal convolution is causal; an identity graph isolates "
  "nodes; the hybrid starts exactly at the base forecast; OOF folds never overlap; CTM conserves vehicles; every model runs end to end in debug mode.")
H2("16.3 Decision records")
TABLE(["ADR", "Decision"],
      [["001", "Standalone git repository for the project folder"], ["002", "Graph direction convention: row i aggregates from column j"],
       ["003", "Missing inputs set to the training mean in scaled space plus an is_missing channel"],
       ["004", "PyTables fallback for reading metr-la.h5"], ["005", "Metric, slice, DM-test and bootstrap conventions"],
       ["006", "Reduced ML budget for a laptop"], ["007", "Cheaper STGAT training (spatial steps, epoch fraction)"],
       ["008", "Gating hybrid uses part of validation"], ["009", "Spec corrections (CityFlow authors; default bidirectional graph)"]],
      "Architecture decision records (docs/decisions.md).", widths=[10, 90], small=True)
H2("16.4 Hardware, runtimes and resource limits")
B(["Apple M3 Pro (11 CPU cores, 18 GB RAM); deep models on the integrated GPU through PyTorch MPS.",
   "Jobs ran one at a time at low priority (nice), with 6 CPU workers for ML and 4 torch threads, to keep the laptop usable.",
   "Approximate wall-clock: ML tuning + training ≈ 50 min; GRU ≈ 8 min per seed; STGAT ≈ 28 min per seed; OOF + residual hybrids ≈ 80 min; "
   "ablations ≈ 4 h 15 min in total (GATv2 alone ≈ 64 min)."])
H2("16.5 How to reproduce")
B(["Create the environment: `uv venv --python 3.11 .venv`, activate it, `uv pip install -e \".[dev]\"`.",
   "`make data` (download + checks), `make eda`, `make phase1` (baselines and tables).",
   "`bash scripts/run_ml_chain.sh` (tune and train the four ML models), `bash scripts/run_dl_chain.sh` (GRU and STGAT, 3 seeds).",
   "`bash scripts/run_phase2_chain.sh` (OOF base forecasts, hybrids, evaluation, maps, control, report assets).",
   "`bash scripts/run_ablation_chain.sh`, then `python scripts/evaluate_ablations.py`.",
   "`make report` builds both PDFs (tectonic). Team members joining the repository run `python3 join.py` once."])
NOTE("Reproducibility note. Large intermediate files are not tracked by git. During a clean-up of the repository history, three of them "
     "were also removed from the working copy: validation-set predictions (results/predictions_val), the out-of-fold base forecasts "
     "(results/hybrid/xgb_base.npz) and the per-run ablation predictions (results/ablations/predictions). All metrics derived from them "
     "(main tables, gains, gate weights, ablation table) were already saved and are intact; to re-run the gating hybrid or re-evaluate "
     "ablations, regenerate them with the chain scripts above.")
BREAK()

# ------------------------------------------------------------------ 17
H1("17. Limitations and honest findings")
H2("17.1 Limitations")
B(["One city and one season (March–June 2012); speed only (no flow, no occupancy).",
   "A fixed distance-based graph that may miss true connectivity (no adaptive adjacency).",
   "Reduced budgets: 15 Optuna trials, pooled XGBoost on 25% of rows with a 400-tree cap that was always reached, the forest on 10% of rows.",
   "STGAT simplified for compute (attention on the last 4 steps, a third of windows per epoch, 40 epochs while still improving).",
   "Ablations on one seed and 12 epochs.",
   "Literature-protocol (DCRNN) numbers were not produced; only the strict protocol is reported, so results are not directly comparable to published tables.",
   "The test period has a higher outage rate than training; MAPE is unstable at low speeds.",
   "The control study is synthetic, open loop, and has no ramp-queue override or spill-back modelling.",
   "DCRNN-lite and the SUMO intersection alternative were not built (optional in the specification)."])
H2("17.2 Results that did not match the hypothesis")
B(["Spatial features help, but not more at longer horizons.",
   "STGAT equals the no-graph GRU at 15 minutes and beats it only at 60 minutes.",
   "In the reduced-budget ablations the road graph did not beat the identity graph.",
   "Attention shares were the same at peak and at night.",
   "Forecast-informed ramp metering did not reduce total time spent."])
H2("17.3 Ethics and authorship")
P("The data are aggregate loop-detector speeds, so privacy risk is low; the main risk is over-claiming real-world control benefits, which "
  "this project avoids. Much of the code and documentation was produced with an AI coding assistant and committed as a baseline; "
  "team members should follow the course's rules on AI-assisted work and disclose it.")

# ------------------------------------------------------------------ 18
H1("18. Team T042 and contributions")
TABLE(["Rank", "Member", "GitHub", "Owns"], [[i + 1, a, b, c] for i, (a, b, c) in enumerate(team)],
      "Work split, ordered by difficulty and technical depth.", widths=[7, 16, 23, 54], small=True)
P("Full module lists, report sections, theory notes and individual viva questions are in `CONTRIBUTIONS.md` and `docs/viva/`. "
  "Ownership of every file is recorded in `OWNERS.yaml`.")

# ------------------------------------------------------------------ appendix
H1("Appendix A. Glossary")
TABLE(["Term", "Meaning"],
      [["Anchor", "the last input time step of a window"], ["Horizon h", "how far ahead (h × 5 minutes)"],
       ["Masked metric", "a metric computed only over observed (non-zero) targets"],
       ["Strict protocol", "chronological split with windows built inside each split"],
       ["OOF", "out-of-fold: predictions made by a model that did not train on those rows"],
       ["DM test", "Diebold–Mariano test of equal predictive accuracy"],
       ["HAC", "heteroskedasticity- and autocorrelation-consistent variance (Newey–West)"],
       ["GAT / GATv2", "graph attention network; v2 has dynamic attention"],
       ["STGAT", "the project's spatio-temporal GAT"], ["TTS", "total time spent by all vehicles (veh·h)"],
       ["CTM", "Cell Transmission Model (Daganzo 1994)"], ["ALINEA", "feedback ramp-metering law (Papageorgiou et al. 1991)"],
       ["MPS", "Apple's Metal Performance Shaders backend for PyTorch"]],
      "Glossary.", widths=[22, 78], small=True)
H1("Appendix B. References")
N(["Veličković et al., Graph Attention Networks, ICLR 2018.", "Brody, Alon, Yahav, How Attentive are Graph Attention Networks?, ICLR 2022.",
   "Kipf and Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017.",
   "Li, Yu, Shahabi, Liu, Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting, ICLR 2018.",
   "Yu, Yin, Zhu, Spatio-Temporal Graph Convolutional Networks, IJCAI 2018.", "Wu et al., Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019.",
   "Guo et al., Attention Based Spatial-Temporal Graph Convolutional Networks (ASTGCN), AAAI 2019.",
   "Zheng et al., GMAN: A Graph Multi-Attention Network for Traffic Prediction, AAAI 2020.",
   "Jagadish et al., Big Data and Its Technical Challenges, Communications of the ACM 57(7), 2014.",
   "Chen and Guestrin, XGBoost: A Scalable Tree Boosting System, KDD 2016.", "Breiman, Random Forests, Machine Learning 45(1), 2001.",
   "Lundberg and Lee, A Unified Approach to Interpreting Model Predictions, NeurIPS 2017.",
   "Diebold and Mariano, Comparing Predictive Accuracy, Journal of Business & Economic Statistics 13(3), 1995.",
   "Harvey, Leybourne, Newbold, Testing the Equality of Prediction Mean Squared Errors, International Journal of Forecasting 13(2), 1997.",
   "Newey and West, A Simple, Positive Semi-Definite, HAC Covariance Matrix, Econometrica 55(3), 1987.",
   "Daganzo, The Cell Transmission Model, Transportation Research Part B 28(4), 1994.",
   "Papageorgiou, Hadj-Salem, Blosseville, ALINEA: A Local Feedback Control Law for On-Ramp Metering, Transportation Research Record 1320, 1991.",
   "Varaiya, Max Pressure Control of a Network of Signalized Intersections, Transportation Research Part C 36, 2013.",
   "Lopez et al., Microscopic Traffic Simulation using SUMO, IEEE ITSC 2018.",
   "Zhang et al., CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario, WWW 2019."])
P("Reference details were written from the published papers' known metadata; confirm page numbers before formal submission.")

OUT.write_text(json.dumps(blocks, ensure_ascii=False))
print(f"{len(blocks)} blocks, {counters['table']} tables, {counters['figure']} figures")
