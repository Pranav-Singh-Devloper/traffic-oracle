# PROJECT_SPEC.md — Traffic Flow Prediction & Simulated Signal Control
## Graph Attention Networks vs. Segment-Level ML on METR-LA

> **Instructions to Claude Code (read first)**
> 1. Build this project **milestone by milestone** (Section 14). At the end of each milestone: run tests, run the milestone's acceptance checks, commit, print a short summary, and **stop for human review**.
> 2. **Never fabricate results.** Every number in the README, report, or slides must be produced by a script and read from `results/`. If something fails, report the failure. Do not invent plausible numbers.
> 3. **Never leak the future.** Splits are chronological. Scalers, imputers, and encoders are fit on training timestamps only. No random shuffling across time boundaries. If you are unsure whether something leaks, ask.
> 4. Signal control is **simulation only**. No code, doc, or slide may claim live or real-world traffic control.
> 5. Prefer clear, readable, well-commented code over cleverness. Every team member must be able to explain every file in an individual viva.
> 6. Keep the code CPU-runnable for small configs (`--debug` flag) and GPU-accelerated for full runs. Auto-detect the device.
> 7. When a design decision in this spec turns out to be wrong in practice, explain why in `docs/decisions.md` (ADR style) before deviating.

---

## 1. Project Summary

**Title:** Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models

**Goal:** Forecast traffic speed (congestion) on the METR-LA road sensor network for 5–60 minutes ahead, and test rigorously whether **network-aware graph learning** improves on **independent per-segment forecasts**. Then run an **optional, simulated** control experiment showing how forecasts could inform a control policy.

**Core hypothesis:** Congestion propagates along upstream and downstream links, typically as shockwaves that travel upstream. Per-segment models see only their own history and cannot represent this propagation directly. A graph model that aggregates information from neighboring sensors should therefore improve accuracy, mostly at **longer horizons** and during **peak periods**, when propagation effects dominate.

**Two evaluation phases:**
- **Phase 1 (ML):** EDA, feature engineering, naive baselines, XGBoost (per-segment and global), multi-output Random Forest, and a spatially augmented XGBoost.
- **Phase 2 (DL + Hybrid):** a GAT-based spatio-temporal model, a no-graph DL ablation, an optional diffusion model, a **justified ML+DL hybrid**, ablations, congestion maps, and the optional simulated control experiment.

---

## 2. Important Data Realities (must be stated in the report)

1. **METR-LA has speed only** (mph). There is **no flow and no occupancy**. Per-segment features for METR-LA are therefore speed-derived. Flow and occupancy features belong to the **PeMS extension** (PEMS04/PEMS08 contain flow, occupancy, and speed). Implement the feature pipeline so it can accept additional channels when PeMS is added.
2. **Missing data is encoded as 0.** In METR-LA, a speed of 0 almost always means sensor outage, not stopped traffic. Treat `speed == 0` as missing and **mask** it in losses and metrics. Never treat it as true zero speed.
3. **METR-LA sensors are on LA County freeways (loop detectors).** Freeways have no traffic signals. The realistic control analogue is **ramp metering**, which is what the main simulated control experiment uses (Section 12).
4. **Data facts** (verify on load and assert in tests):
   - 207 sensors, 5-minute intervals.
   - 34,272 timestamps, 2012-03-01 00:00 to 2012-06-27 23:55 (119 days).
   - `metr-la.h5` is a pandas HDF5 file, read with `pd.read_hdf(path)` (key `df`). Columns are sensor IDs and the index is a DatetimeIndex.
   - `adj_mx.pkl` is a pickle tuple `(sensor_ids, sensor_id_to_ind, adj_mx)`. In Python 3, load it with `pickle.load(f, encoding="latin1")`.
   - Sensor coordinates are in `graph_sensor_locations.csv` and pairwise road distances in `distances_la_2012.csv` (DCRNN repository, `data/sensor_graph/`).

---

## 3. Data Acquisition

Create `scripts/download_data.py` that:
1. Tries to fetch the METR-LA files from known public mirrors: the DCRNN repository (`liyaguang/DCRNN`, where `data/sensor_graph/` holds `adj_mx.pkl`, `distances_la_2012.csv`, `graph_sensor_ids.txt`, `graph_sensor_locations.csv`, and the README links to `metr-la.h5`), the PyTorch Geometric Temporal METR-LA loader, and Kaggle mirrors.
2. If automatic download fails, prints clear instructions telling the user where to place the files manually:
   ```
   data/raw/metr-la.h5
   data/raw/adj_mx.pkl
   data/raw/distances_la_2012.csv
   data/raw/graph_sensor_ids.txt
   data/raw/graph_sensor_locations.csv
   ```
3. Verifies file integrity with shape checks (Section 2.4) and writes `data/raw/MANIFEST.json` containing file sizes and SHA256 hashes.

`data/` is git-ignored. Only `MANIFEST.json` and a `data/README.md` describing provenance are committed.

---

## 4. Repository Structure

```
traffic-gnn-forecasting/
├── README.md                     # overview, results table (auto-generated), how to run
├── PROJECT_SPEC.md               # this file
├── CONTRIBUTIONS.md              # who did what (per member, per module)
├── LICENSE
├── pyproject.toml / requirements.txt
├── Makefile                      # make data / eda / phase1 / phase2 / report / test
├── configs/
│   ├── base.yaml                 # paths, seeds, split ratios, horizons, history window
│   ├── ml/xgb_per_segment.yaml
│   ├── ml/xgb_global.yaml
│   ├── ml/rf_multioutput.yaml
│   ├── ml/xgb_spatial.yaml
│   ├── dl/gru_nograph.yaml
│   ├── dl/stgat.yaml
│   ├── dl/dcrnn_lite.yaml        # optional
│   ├── hybrid/residual_hybrid.yaml
│   └── ablations/*.yaml
├── data/                         # git-ignored (except README + MANIFEST)
├── src/traffic/
│   ├── __init__.py
│   ├── config.py                 # YAML loading + dataclass validation
│   ├── seed.py                   # global seeding, deterministic flags
│   ├── data/
│   │   ├── load.py               # read h5, pkl, csvs; integrity asserts
│   │   ├── graph.py              # build adjacency from distances; compare with adj_mx.pkl
│   │   ├── missing.py            # mask construction, outage stats, input imputation
│   │   ├── split.py              # chronological split by timestamp
│   │   ├── scaling.py            # masked StandardScaler fit on train only
│   │   ├── windows.py            # sliding windows (history H, horizon F) within each split
│   │   └── datasets.py           # torch Dataset/DataLoader for DL
│   ├── features/
│   │   ├── temporal.py           # cyclic time features, weekend, holiday
│   │   ├── lags.py               # lagged speeds, rolling stats, diffs
│   │   ├── spatial.py            # neighbor-lag features via adjacency (for XGB-spatial)
│   │   └── build_tabular.py      # assemble tabular design matrices for ML
│   ├── models/
│   │   ├── baselines.py          # last-value (persistence), historical average, seasonal naive
│   │   ├── ml_models.py          # XGB per-segment, XGB global, RF multi-output
│   │   ├── layers/gat_dense.py   # own dense masked GAT layer (explainable)
│   │   ├── layers/diffusion.py   # optional diffusion conv
│   │   ├── gru_nograph.py        # shared-weight GRU per node, no graph (DL ablation)
│   │   ├── stgat.py              # main spatio-temporal GAT model
│   │   ├── dcrnn_lite.py         # optional
│   │   └── hybrid.py             # residual hybrid (ML base + graph residual)
│   ├── training/
│   │   ├── losses.py             # masked MAE / MSE / Huber
│   │   ├── trainer.py            # train loop, early stopping, checkpointing, logging
│   │   └── tuning.py             # Optuna on validation only
│   ├── evaluation/
│   │   ├── metrics.py            # masked MAE, RMSE, MAPE
│   │   ├── slices.py             # horizon, peak/off-peak, weekday/weekend, per sensor
│   │   ├── stats_tests.py        # Diebold–Mariano, bootstrap CIs
│   │   └── report_tables.py      # CSV → Markdown + LaTeX tables
│   ├── viz/
│   │   ├── eda_plots.py
│   │   ├── congestion_map.py     # folium HTML + static matplotlib maps
│   │   ├── attention_viz.py      # GAT attention on map
│   │   └── horizon_plots.py
│   └── control/                  # optional simulated control
│       ├── ctm.py                # cell transmission model freeway simulator (pure Python)
│       ├── alinea.py             # reactive ALINEA ramp metering
│       ├── predictive_alinea.py  # forecast-informed ramp metering
│       ├── sumo/                 # optional SUMO network files + TraCI runner
│       └── run_experiment.py
├── scripts/
│   ├── download_data.py
│   ├── run_eda.py
│   ├── train_ml.py               # --model xgb_per_segment|xgb_global|rf|xgb_spatial
│   ├── train_dl.py               # --model gru_nograph|stgat|dcrnn_lite
│   ├── train_hybrid.py
│   ├── run_ablations.py
│   ├── evaluate_all.py           # builds results/metrics_*.csv from saved predictions
│   ├── make_maps.py
│   └── make_report_assets.py     # figures + LaTeX tables into report/
├── notebooks/                    # exploration only; final logic lives in src/
│   ├── 01_eda.ipynb
│   ├── 02_ml_results.ipynb
│   └── 03_dl_results.ipynb
├── results/
│   ├── predictions/              # .npz: y_true, y_pred, mask, timestamps per model (git-ignored if large)
│   ├── metrics/                  # CSVs (committed)
│   ├── figures/                  # PNG/PDF (committed)
│   └── maps/                     # HTML maps
├── report/
│   ├── phase1/main.tex           # IEEEtran
│   ├── phase2/main.tex
│   ├── refs.bib
│   └── figures/ tables/          # auto-generated
├── slides/                       # presentation sources
├── docs/
│   ├── decisions.md              # ADRs
│   ├── theory/                   # derivations for viva (Section 15)
│   ├── viva/                     # per-member Q&A prep
│   └── experiment_log.md         # auto-appended run log
└── tests/
    ├── test_load.py
    ├── test_graph.py
    ├── test_split_no_leakage.py
    ├── test_scaling_train_only.py
    ├── test_windows.py
    ├── test_metrics_masking.py
    ├── test_gat_layer.py         # own GAT vs PyG GATConv equivalence
    └── test_end_to_end_debug.py  # tiny config runs every model for 1 epoch
```

**Stack:** Python 3.11, pandas, numpy, scikit-learn, xgboost, PyTorch 2.x, PyTorch Geometric, NetworkX, folium, matplotlib, optuna, pyyaml, tables (for HDF5), pytest, ruff, black, optional `sumo` + `traci`. Pin versions in `requirements.txt`. Use MLflow or plain CSV/JSON logging, kept simple. Use plain CSV logging plus `docs/experiment_log.md` unless the team asks for MLflow.

---

## 5. Data Pipeline (shared by ML and DL)

### 5.1 Global settings (`configs/base.yaml`)
```yaml
seed: 42
interval_minutes: 5
history_steps: 12          # H = 60 min of history (equal for ALL models)
horizon_steps: 12          # F = predict 5..60 min ahead
report_horizons: [3, 6, 12]  # 15, 30, 60 min (literature standard)
split: {train: 0.7, val: 0.1, test: 0.2}   # chronological, by timestamp
missing_value: 0.0
peak_windows:              # LA local time, weekdays only, evaluated on TARGET timestamp
  am: ["06:00", "10:00"]
  pm: ["15:00", "19:00"]
congestion_threshold_mph: 40
```

### 5.2 Missing data
- Build `mask = (speed > 0)` with shape `[T, N]`.
- Report in EDA: overall missing rate, per-sensor missing rate, longest outage runs, and outage clustering in time (whole-network outages versus single sensors).
- **Targets:** never imputed. Masked in loss and metrics.
- **Inputs:** two options, controlled by config. The default (a) forward-fills outages up to 6 steps (30 min), fills the remainder with the training-set mean of that sensor at that time-of-day slot, and adds a binary `is_missing` input channel. Option (b) leaves zeros as they are with the mask channel, matching the DCRNN convention. Option (a) is the default; (b) is an ablation.
- Imputation statistics (time-of-day means) are computed on **training timestamps only**.

### 5.3 Chronological split
- Split the **timestamp axis**: first 70% train, next 10% val, last 20% test.
- **Primary (strict) protocol:** windows are built **inside** each split, so no window's input or target crosses a split boundary.
- **Literature-comparable protocol (optional, clearly labeled):** the DCRNN convention, which builds all windows first and then splits samples 70/10/20. Report this only for comparison with published numbers.
- `tests/test_split_no_leakage.py` asserts `max(train_target_ts) < min(val_input_ts)` and `max(val_target_ts) < min(test_input_ts)` under the strict protocol.

### 5.4 Scaling
- Masked z-score: mean and std computed from **training timestamps, observed values only** (zeros excluded). Use one global scaler, with a per-sensor scaler as an ablation.
- Models train in scaled space. **All metrics are computed in original mph** after inverse transform.
- `tests/test_scaling_train_only.py` asserts that changing val/test values does not change the scaler parameters.

### 5.5 Time features (cyclic)
- `tod_sin, tod_cos` = sin/cos(2π · minute_of_day / 1440)
- `dow_sin, dow_cos` = sin/cos(2π · day_of_week / 7)
- `is_weekend`
- `is_holiday` (US federal holidays in range: Memorial Day 2012-05-28; use the `holidays` package)
- Explain in docs why cyclic encoding is used: 23:55 and 00:00 should be close in feature space.

### 5.6 Graph construction (`src/traffic/data/graph.py`)
- Rebuild the adjacency **from `distances_la_2012.csv`** using the thresholded Gaussian kernel (DCRNN):
  `W_ij = exp(-(dist(v_i, v_j)/σ)²)` if that value is ≥ κ, else 0, with σ = std of all finite distances and κ = 0.1.
- The road distance is **directed**, so `W` is asymmetric. Document the message direction convention explicitly in `docs/decisions.md` (for example, "`W[i, j] > 0` means sensor j's information flows to i").
- **Verify** that the rebuilt matrix matches `adj_mx.pkl` (assert max abs difference < 1e-6 or document any differences). Use `adj_mx.pkl` as the canonical matrix and keep the rebuild as proof of understanding.
- Build a NetworkX `DiGraph` and report node count, edge count, in/out-degree distribution, weakly and strongly connected components, and diameter of the largest component.
- Variants for ablations: `forward` (W), `backward` (Wᵀ), `bidirectional` (W + Wᵀ), `identity` (no neighbors), `random` (same edge count, random edges), `knn_geo` (k nearest by lat/lon).

### 5.7 Sliding windows
For each split, sample `t` yields:
- `X[t] = speed[t-H+1 : t+1]` → `[H, N, C]` (C channels: scaled speed, is_missing, tod_sin, tod_cos, dow_sin, dow_cos, …)
- `Y[t] = speed[t+1 : t+F+1]` → `[F, N]` (original or scaled, with `Ymask` alongside)
- Also store `target_timestamps[t]` for slicing peak and off-peak.

---

## 6. Exploratory Data Analysis (Phase 1, `scripts/run_eda.py`)

Produce figures in `results/figures/eda/` and a summary in `docs/eda_summary.md`. Each item below is one figure plus two or three sentences of interpretation:

1. Missingness heatmap (sensor × day) and per-sensor missing-rate histogram.
2. Average daily speed profile (weekday vs weekend) with IQR bands.
3. Heatmap of mean speed by (time-of-day × day-of-week).
4. Speed distribution overall and at peak vs off-peak.
5. Autocorrelation (ACF/PACF) for a few representative sensors, showing daily seasonality at lag 288.
6. **Spatial correlation vs graph distance:** correlation between sensor pairs plotted against shortest-path hops, as evidence that neighbors carry information.
7. **Lagged cross-correlation upstream → downstream:** for selected adjacent sensor pairs, plot cross-correlation vs lag and show peak correlation at non-zero lag (evidence of propagation). This motivates the graph model and is a key plot for the "identify the gap" argument.
8. Sensor map (folium) colored by mean speed and by missing rate.
9. Congestion frequency map: share of time each sensor is below 40 mph during peaks.
10. Train/val/test period visualization on a timeline, to check that the splits contain comparable weekday and peak mixes and to note any distribution shift.

---

## 7. Phase 1 — Machine Learning

### 7.1 Naive baselines (`models/baselines.py`)
- **Persistence (last value):** ŷ(t+h) = y(t).
- **Historical average:** mean speed of that sensor at the same (day-of-week, time-of-day) slot, computed on train only.
- **Seasonal naive (daily):** ŷ(t+h) = y(t+h−288).
- These baselines are essential because a model that does not beat persistence at 15 min is not useful.

### 7.2 Tabular features (`features/`)
Per (sensor i, time t) row:
- Lags: `speed_{t}, speed_{t-1}, …, speed_{t-11}` (equal history window H = 12)
- Rolling mean, std, min, and max over 3, 6, and 12 steps
- First differences `speed_t − speed_{t-1}`, `speed_t − speed_{t-3}`
- `is_missing` flags for the lags and the count of missing values in the window
- Cyclic time features (5.5), `is_weekend`, `is_holiday`
- Same-time-yesterday speed `speed_{t-288}`. **Note:** this uses data outside the 12-step window. Make it an explicit config flag (`use_daily_lag`), keep it **off** in the main equal-window comparison, and evaluate it as a separate experiment.
- Static sensor features (for global models): sensor ID (categorical or target-encoded on train only), lat, lon, in-degree, out-degree
- PeMS extension only: flow and occupancy lags, along with flow/occupancy ratios.

### 7.3 Models
1. **XGBoost per segment** (`xgb_per_segment`): one model per sensor per reported horizon {3, 6, 12} by default. That is 207 × 3 = 621 models with `tree_method="hist"`, parallelized with joblib. A config flag trains all 12 horizons (2,484 models) if time allows.
2. **XGBoost global** (`xgb_global`): one pooled model across sensors with sensor ID and static features, trained per horizon (12 models). This model shares statistical strength across sensors but still has no neighbor information.
3. **Random Forest multi-output** (`rf_multioutput`): `RandomForestRegressor` natively predicts all 12 horizons jointly. Train global (pooled) by default, with per-segment training as an option. Subsample training rows if memory is tight and document it.
4. **XGBoost spatial** (`xgb_spatial`): `xgb_global` plus neighbor features. For sensor i: adjacency-weighted mean of neighbors' last 3 lags (upstream and downstream separately), 2-hop aggregates, and the difference between own speed and neighbor mean. This is the **bridge experiment**: it tests how much of the graph benefit a hand-engineered spatial feature can capture before deep learning is introduced.

### 7.4 Training protocol
- Missing targets are dropped from ML training rows (equivalent to masking).
- Tune hyperparameters with Optuna on the **validation split only**, with a fixed trial budget per model (for example 40 trials) logged in `results/metrics/tuning_*.csv`.
- Early stopping on validation for XGBoost.
- Retrain on train only (not train+val) for the final test evaluation so the protocol matches the DL models. Document this choice.
- Save predictions in the common format: `results/predictions/{model}.npz` with `y_pred [S, F, N]`, `y_true`, `mask`, `target_ts`.

### 7.5 Phase 1 analysis
- Feature importance: XGBoost gain plus **SHAP** on a sample. Discuss which lags and time features matter, and whether neighbor features matter in `xgb_spatial`.
- Error vs horizon curves for all ML models.
- Per-sensor error map: which locations are hardest?
- **The gap argument:** compare `xgb_global` with `xgb_spatial` by horizon and peak/off-peak. If spatial features help mostly at long horizons and peaks, that supports the propagation hypothesis and motivates Phase 2. If they barely help, report that honestly.

---

## 8. Evaluation Protocol (used identically for all models)

### 8.1 Metrics (`evaluation/metrics.py`), computed in mph and masked
- `masked_MAE = Σ m·|y−ŷ| / Σ m`
- `masked_RMSE = sqrt(Σ m·(y−ŷ)² / Σ m)`
- `masked_MAPE = Σ m'·|y−ŷ|/y / Σ m'` · 100, with `m' = m ∧ (y > 0)` (optionally also y ≥ 1 mph; document the choice)
- Unit tests in `test_metrics_masking.py` must use hand-computed examples, including all-masked edge cases.

### 8.2 Slices (`evaluation/slices.py`)
Each metric is reported for:
- **Every horizon 1..12**, with headline tables at 3, 6, and 12 (15, 30, 60 min)
- **Peak vs off-peak** (weekday peak windows from config, judged on the **target** timestamp), with weekend as its own slice
- **Congested vs free-flow targets** (y < 40 mph vs ≥ 40)
- Per sensor (for maps)

### 8.3 Fairness rules
- Same chronological split, same test timestamps, and same masks for every model.
- Same history window H = 12 for every model in the main table. A separate **history-window sweep** H ∈ {6, 12, 24} is run for the best ML model and the best DL model.
- Comparable tuning budgets, documented.
- DL models: report mean ± std over **3 seeds**.

### 8.4 Statistical significance (`evaluation/stats_tests.py`)
- **Diebold–Mariano test** on per-timestamp absolute errors (averaged over sensors) between pairs: STGAT vs XGB-global, STGAT vs GRU-nograph, and Hybrid vs each component, per headline horizon. Use a HAC variance estimate because errors are autocorrelated.
- **Block bootstrap** (block = 1 day) 95% CIs for MAE per horizon.

### 8.5 Outputs
- `results/metrics/main_table.csv`: model × horizon × metric.
- `results/metrics/peak_offpeak.csv`, `results/metrics/significance.csv`.
- `scripts/make_report_assets.py` converts these to LaTeX (`booktabs`) and Markdown, and inserts the Markdown table into the README between `<!-- RESULTS_START -->` and `<!-- RESULTS_END -->`.

### 8.6 Sanity bounds
Published METR-LA results for strong graph models (for example DCRNN and Graph WaveNet) report MAE of roughly 2.7 at 15 min, 3.1 at 30 min, and 3.5 at 60 min under the DCRNN protocol. Under the strict protocol, numbers may differ slightly.
- If any model reports **MAE@60min below about 3.0**, investigate for leakage before celebrating.
- If STGAT does not beat persistence at 15 min, something is broken.
- Print these checks automatically in `evaluate_all.py` as warnings.

---

## 9. Phase 2 — Deep Learning

### 9.1 Models
**(a) `gru_nograph`: no-graph DL ablation (mandatory).** A GRU encoder shared across nodes processes each sensor's history independently, followed by an MLP head outputting F horizons. Inputs and capacity are the same as STGAT minus the spatial layers. This is the fair test of whether the **graph** helps, as opposed to deep learning in general.

**(b) `stgat`: main model, a spatio-temporal Graph Attention Network.**
```
Input X: [B, H, N, C]
 → Linear embed C→d (d=64)
 → + learnable node embedding E ∈ R^{N×d_e} (ablation: off)
 → L=2 ST blocks, each:
      Temporal: gated temporal conv (kernel 3, dilation 1/2) OR GRU over H, per node
      Spatial:  multi-head GAT over the road graph at each time step (K=4 heads, concat → d)
      Residual connection + LayerNorm + dropout
 → Temporal readout (last step or attention pooling over H)
 → MLP decoder → [B, F, N]   (non-autoregressive: all 12 horizons at once)
```
- **GAT layer (`layers/gat_dense.py`).** Implement the dense masked GAT yourself for explainability. N = 207 is small, so dense [N×N] attention masked by adjacency is fast and batches easily:
  - `h'_i = W h_i`
  - `e_ij = LeakyReLU(aᵀ [h'_i || h'_j])` for j ∈ N(i) ∪ {i}, otherwise −∞
  - `α_ij = softmax_j(e_ij)`, then `h_i^out = ELU(Σ_j α_ij h'_j)`, with K heads concatenated (averaged in the last layer if used as output)
  - Optional: add the edge weight `W_ij` as a bias in `e_ij` (ablation)
  - Option to switch to GATv2 (`a ᵀ LeakyReLU(W[h_i||h_j])`) as an ablation
- `tests/test_gat_layer.py` checks numerical equivalence (within tolerance) with `torch_geometric.nn.GATConv` on a small random graph with copied weights.
- Return attention weights for visualization.

**(c) `dcrnn_lite` (optional).** Diffusion convolution (K-step random walks on W and Wᵀ) inside a GRU cell, encoder-decoder. Use it as the "diffusion graph model" mentioned in the brief. If time is short, use `torch_geometric_temporal.nn.recurrent.DCRNN` and document that choice.

### 9.2 Training
- Loss: masked MAE in scaled space (optionally Huber), matching the evaluation metric.
- Adam, lr 1e-3, weight decay 1e-4, ReduceLROnPlateau or cosine schedule, gradient clipping at 5.0, batch size 64, max 100 epochs, early stopping on **val masked MAE** with patience 15, best checkpoint restored.
- Regularization (graded under "DL Dataset & Regularization"): dropout (0.1–0.3), weight decay, early stopping, attention dropout, and optionally DropEdge. Each one is justified in the report and appears in at least one ablation.
- Optional data augmentation, train split only: Gaussian noise on inputs and random sensor masking (simulated outages) to improve robustness. Evaluate with an ablation.
- Run on 3 seeds and log train/val curves to `results/figures/training/`.
- Mixed precision on GPU. A `--debug` config uses 10 sensors, 2 days, and 1 epoch for CI.

### 9.3 Interpretability
- Plot GAT attention weights on the map for selected sensors during AM peak vs night. Do upstream sensors receive more attention at peak? Present this as a qualitative observation, not proof of causality.
- Show error vs horizon curves for all models on one plot (the key figure of the project).

---

## 10. Hybrid Approach (Phase 2 bonus, must be technically justified)

### 10.1 Primary hybrid: **Residual Graph Correction (XGB → STGAT residual)**
**Rationale:**
- Per-segment ML (XGBoost) is strong at local autoregressive structure and calendar effects, and is cheap and stable.
- It cannot model spatial propagation (the identified gap).
- So the GNN is trained to learn **only what XGBoost misses**: the residual `r = y − ŷ_XGB`. Spatial propagation should be a large part of that residual, especially at peak and long horizons.

**Formulation:** `ŷ_hybrid(t+h, i) = ŷ_XGB(t+h, i) + g_θ(X_window, ŷ_XGB, G)(h, i)`, where `g_θ` is STGAT with ŷ_XGB added as an extra input channel to the decoder.

**Leakage-safe training (critical; explain in the viva):**
1. XGBoost predictions on its **own training rows** are overfit, so their residuals are unrealistically small. Do not use them.
2. Generate **out-of-fold (OOF)** XGBoost predictions on the train period with **forward-chaining time-series CV**. Split train into 5 chronological blocks. For block k, train on blocks < k and predict block k. Block 1 has no OOF prediction and is dropped from residual training, or a smaller warm-up model is used; document the choice.
3. Train the final XGBoost on the full train split and use it to predict val and test.
4. Train `g_θ` on OOF residuals (train) and early-stop on val residual MAE.
5. Evaluate `ŷ_hybrid` on test with the standard protocol.

### 10.2 Alternative hybrid (ablation): **Graph-informed gating ensemble**
A small learned gate `w(h, peak_flag, tod, missing_rate)` ∈ [0, 1], trained on **validation predictions only**, mixes XGB and STGAT: `ŷ = w·ŷ_STGAT + (1−w)·ŷ_XGB`. Report whether the gate learns to trust the graph model more at long horizons and peaks. If the gate is trained on val, then val cannot also be used for model selection. Hold out the last 20% of val for gate validation and document this.

### 10.3 Required hybrid evidence
- The hybrid must be compared with **both** components and with the simple average `0.5·XGB + 0.5·STGAT`.
- Diebold–Mariano significance vs each component.
- Analyze where the gain comes from (horizon, peak, congested targets).
- If the hybrid does **not** beat STGAT, report that honestly and explain why (for example, STGAT already captures the local structure). An honest negative result with correct methodology is still technically sound.

---

## 11. Ablation Studies (`scripts/run_ablations.py`)

Each ablation changes **one thing** from the STGAT default and is run on 3 seeds. Results go to `results/metrics/ablations.csv` with a bar chart of MAE@15/30/60.

| # | Ablation | Question answered |
|---|----------|-------------------|
| A1 | Graph = identity (no neighbors) | Does spatial information help at all? |
| A2 | Graph = random (same edge count) | Is it the *road* topology or just extra parameters? |
| A3 | Forward vs backward vs bidirectional adjacency | Does propagation direction matter? |
| A4 | GAT vs GATv2 vs mean aggregation (GCN-style) | Does *attention* help over fixed weights? |
| A5 | Heads K ∈ {1, 2, 4, 8} | Multi-head benefit |
| A6 | Layers L ∈ {1, 2, 3} | Receptive field vs over-smoothing |
| A7 | No node embedding | Value of sensor identity |
| A8 | No time features | Value of calendar context |
| A9 | Zero-fill input vs imputation + mask channel | Missing-data handling |
| A10 | No dropout / no weight decay | Regularization effect |
| A11 | History H ∈ {6, 12, 24} | Window sensitivity (also run for XGB) |
| A12 | Edge weight bias in attention on/off | Distance prior |
| A13 | Hybrid: residual vs gating vs average | Hybrid design choice |

---

## 12. Simulated Control Experiment (optional, Phase 2)

> **Disclaimer to include everywhere:** "This is an offline simulation on a synthetic network. It demonstrates how forecasts *could* inform a control policy. No live or real-world traffic control is performed or claimed."

### 12.1 Primary: Freeway ramp metering with a Cell Transmission Model (pure Python, no SUMO needed)
- `control/ctm.py` implements Daganzo's Cell Transmission Model for a freeway corridor of about 10–15 cells with 2–3 on-ramps and a triangular fundamental diagram (free-flow speed about 65 mph, capacity, jam density).
- Mainline demand profiles are derived from METR-LA by mapping a chosen corridor of adjacent sensors along one freeway to cells and converting speed to a demand profile via the fundamental diagram. Document all assumptions, since speed-to-flow inversion is approximate.
- Policies compared:
  1. **No control** (ramps unmetered)
  2. **Fixed-time metering**
  3. **ALINEA** (reactive feedback: `r(k) = r(k−1) + K_R·(o* − o(k))`)
  4. **Predictive ALINEA**: same control law, but driven by the **STGAT 15-min forecast** of downstream speed, with density inferred from the fundamental diagram, so it acts before the breakdown
  5. **Oracle-predictive** (uses true future values, as an upper bound)
- Metrics: total time spent (TTS), total travel distance, mainline average speed, max ramp queue length, and duration of congestion.
- Run multiple demand days (for example 5 test-set weekdays) and report mean ± std.

### 12.2 Alternative: SUMO intersection grid (if the team wants "signal" control literally)
- A small synthetic grid (for example 3×3) in SUMO with demand shaped by METR-LA temporal profiles.
- Policies: fixed-time, actuated, **max-pressure**, and forecast-aware max-pressure (pressure computed on forecast queue proxies), via TraCI.
- Metrics: average delay, queue length, throughput, number of stops.
- CityFlow (Zheng et al., 2019) is an acceptable lighter-weight alternative simulator for this option.

---

## 13. Congestion Maps & Visualizations (`viz/`, `scripts/make_maps.py`)
1. **Interactive folium maps** (`results/maps/*.html`): sensors colored by speed (red < 40 mph, orange 40–55, green > 55) for (a) ground truth, (b) STGAT forecast, and (c) XGB forecast at +15/+30/+60 min, for a chosen AM-peak timestamp in the test set. Add a time slider (folium `TimestampedGeoJson`) for an animated peak hour.
2. **Error maps:** per-sensor MAE difference (XGB − STGAT), diverging colormap. Where does the graph help most?
3. **Static PNG/PDF versions** of the key maps for the report (matplotlib + contextily basemap, optional).
4. **Space-time diagram** for one freeway corridor (sensor order along the freeway × time, colored by speed), with truth vs forecast side by side. This makes upstream-propagating shockwaves visible and is ideal for the presentation.
5. Error vs horizon line plot (all models), peak vs off-peak grouped bars, and the ablation bar chart.

---

## 14. Milestones (Claude Code: stop after each for review)

### M0 — Scaffold
Repo structure, `requirements.txt`, Makefile, configs, seeding, ruff/black, pytest, GitHub Actions CI running `pytest` + `--debug` end-to-end, plus `README` skeleton, `CONTRIBUTIONS.md`, and `docs/decisions.md` templates.
**Accept when:** `make test` passes on an empty pipeline and CI is green.

### M1 — Data + Graph
Download script, loaders, integrity asserts, adjacency rebuild + verification, NetworkX stats, missing mask, chronological split, train-only scaler, windows, and the leakage and scaling tests.
**Accept when:** all data tests pass, graph rebuild matches `adj_mx.pkl` (or the difference is documented), and split boundaries are printed.

### M2 — EDA
All Section 6 figures plus `docs/eda_summary.md` with interpretation, including the cross-correlation propagation evidence.
**Accept when:** figures are generated by `make eda` and the summary is written.

### M3 — Baselines + Evaluation framework
Persistence, historical average, seasonal naive; metrics, slices, significance tests, table generation.
**Accept when:** `results/metrics/main_table.csv` contains the baselines and the metric unit tests pass.

### M4 — Phase 1 ML
Tabular features, XGB per-segment, XGB global, RF multi-output, XGB spatial, Optuna tuning, SHAP, ML error analysis, and the "gap" analysis.
**Accept when:** all ML models appear in the main table with peak/off-peak slices, and the SHAP + gap figures exist.

### M5 — Phase 1 Report & Slides
`report/phase1/main.tex` (IEEEtran) with literature review, dataset & EDA, feature engineering, theory (XGBoost objective, RF variance reduction, why chronological splits), results, and limitations. Slides outline in `slides/phase1.md`.
**Accept when:** the LaTeX compiles (`latexmk -pdf`) and every number in it comes from `report/tables/` generated files.

### M6 — DL models
Dense GAT layer + equivalence test, GRU-nograph, STGAT, trainer with early stopping, 3 seeds; optional DCRNN-lite.
**Accept when:** STGAT beats persistence at all horizons, results exist for 3 seeds, and training curves are saved.

### M7 — Hybrid
OOF XGBoost predictions with forward-chaining CV, residual hybrid, gating hybrid, simple average, and significance tests.
**Accept when:** hybrid rows appear in the main table with DM tests vs both components.

### M8 — Ablations
All of Section 11, as a table and figure.

### M9 — Maps & Interpretability
Folium maps, error maps, space-time diagram, attention visualization.

### M10 — Simulated control (optional)
CTM + ALINEA + predictive ALINEA (+ oracle), with results table and disclaimer. SUMO grid only if requested.

### M11 — Phase 2 Report, Slides, Viva prep
`report/phase2/main.tex`, `slides/phase2.md`, the `docs/theory/` notes, and `docs/viva/` per-member Q&A. Final README with reproduction commands and the results table.

---

## 15. Theory Notes for the Viva (`docs/theory/`)
Write one concise Markdown file per topic, with equations, intuition, and "likely viva questions + answers":
1. `chronological_splits.md`: why random splits leak information through autocorrelation and overlapping windows; the strict vs DCRNN protocol.
2. `xgboost.md`: additive trees, second-order Taylor objective, gain formula, regularization (λ, γ), and why per-segment vs global matters.
3. `random_forest.md`: bagging, feature subsampling, variance reduction, native multi-output splitting.
4. `gat.md`: attention coefficients, masked softmax over neighbors, multi-head attention, comparison with GCN (fixed normalized weights) and GATv2 (the static attention limitation), complexity O(|E|·d).
5. `diffusion_conv.md`: random-walk diffusion on a directed graph, bidirectional K-hop filters (DCRNN).
6. `spatiotemporal_modeling.md`: shockwave propagation and why graph receptive field × temporal depth matters; over-smoothing.
7. `missing_data.md`: masking vs imputation, why zero ≠ no traffic, how masked losses produce unbiased gradients.
8. `metrics.md`: MAE vs RMSE (sensitivity to large errors), MAPE pitfalls at low speeds, why results are reported per horizon.
9. `significance_testing.md`: Diebold–Mariano, HAC variance, block bootstrap.
10. `hybrid.md`: residual learning, why OOF predictions are needed, stacking theory.
11. `control.md`: CTM, fundamental diagram, ALINEA, max-pressure, and why it is simulation only.

---

## 16. Team Contribution & Individual Viva Readiness
- `CONTRIBUTIONS.md` contains a table of member → modules owned → experiments run → report sections written. Actual split for Team T042, 5 members (ordered by difficulty; see `CONTRIBUTIONS.md`):
  - **Pranav Singh (rank 1, hardest):** GAT layer, STGAT, GRU-nograph, DL training and trainer, integration/review
  - **Kunal (rank 2):** feature engineering, Phase 1 ML models and tuning, SHAP, gap analysis, out-of-fold hybrids (residual, gating, average)
  - **Himanshu (rank 3):** evaluation framework, baselines, significance tests, hybrid gain analysis, simulated control
  - **Kanchan (rank 4):** data pipeline, graph construction, missing-data handling, EDA, ablation runs
  - **Vaibhav (rank 5, most mechanical):** scaffold/CI, maps, attention plots, README, slides, report assembly, bibliography checks
- **Everyone** must be able to explain: the split protocol, masking, the adjacency construction, the GAT equations, the hybrid OOF logic, and the headline results.
- Commits should come from each member's own account to their owned modules (graders look at GitHub history). Use feature branches and pull requests with short descriptions.
- `docs/viva/<member>.md` contains about 20 likely questions on the member's own modules plus 10 shared questions, with answers.

---

## 17. Report Structure (IEEE, `IEEEtran` class)

**Phase 1 (ML):** Abstract · I. Introduction · II. Related Work (classical, ML, early graph models) · III. Dataset & EDA · IV. Problem Formulation · V. Feature Engineering · VI. Methods (baselines, XGB, RF, XGB-spatial) with theory · VII. Experimental Setup (splits, scaling, tuning, metrics) · VIII. Results (horizon tables, peak/off-peak, SHAP, gap analysis) · IX. Discussion & Limitations · X. Conclusion & Phase 2 Plan · References

**Phase 2 (DL):** Abstract · I. Introduction · II. DL Literature Review · III. Data, Graph & Regularization · IV. Architecture (STGAT, GRU-nograph, DCRNN-lite) · V. Hybrid Method & Rationale · VI. Experimental Setup · VII. Results (main table, significance, horizon curves, peak/off-peak) · VIII. Ablations · IX. Interpretability & Congestion Maps · X. Simulated Control (with disclaimer) · XI. Limitations & Ethics · XII. Conclusion · References

**Limitations to state honestly:** a single city and season (Mar–Jun 2012); speed-only data; a fixed distance-based graph that may miss true connectivity; a simulation gap for control; MAPE instability at low speeds.

---

## 18. Literature (`report/refs.bib`)
Core (from brief):
- Veličković et al., "Graph Attention Networks," ICLR 2018.
- Zheng et al., "CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario," WWW 2019.

Graph-based traffic forecasting:
- Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting," ICLR 2018 (source of METR-LA preprocessing and adjacency).
- Yu, Yin, Zhu, "Spatio-Temporal Graph Convolutional Networks," IJCAI 2018.
- Wu et al., "Graph WaveNet for Deep Spatial-Temporal Graph Modeling," IJCAI 2019.
- Guo et al., "Attention Based Spatial-Temporal Graph Convolutional Networks (ASTGCN)," AAAI 2019.
- Zheng et al., "GMAN: A Graph Multi-Attention Network for Traffic Prediction," AAAI 2020.
- Brody, Alon, Yahav, "How Attentive are Graph Attention Networks?" (GATv2), ICLR 2022.
- Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks," ICLR 2017.
- Jagadish et al., "Big Data and Its Technical Challenges," CACM 2014 (METR-LA data origin as cited by DCRNN).

ML & evaluation:
- Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD 2016.
- Breiman, "Random Forests," Machine Learning 2001.
- Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP), NeurIPS 2017.
- Diebold & Mariano, "Comparing Predictive Accuracy," JBES 1995.

Control & simulation:
- Daganzo, "The Cell Transmission Model," Transportation Research Part B 1994.
- Papageorgiou et al., "ALINEA: A Local Feedback Control Law for On-Ramp Metering," Transportation Research Record 1991.
- Varaiya, "Max Pressure Control of a Network of Signalized Intersections," Transportation Research Part C 2013.
- Lopez et al., "Microscopic Traffic Simulation using SUMO," IEEE ITSC 2018.

Claude Code: verify each BibTeX entry's details (venue, year, pages) against the actual paper before finalizing. Do not invent DOIs.

---

## 19. Code Quality Requirements (graded)
- Type hints on public functions, docstrings (NumPy style) explaining **why** as well as what.
- No logic only in notebooks; notebooks import from `src/`.
- `ruff` + `black` clean; pre-commit hooks.
- Deterministic seeds; log config + git commit hash with every run in `docs/experiment_log.md`.
- Tests: data integrity, no leakage, train-only scaling, masked metrics, GAT equivalence, and an end-to-end debug run.
- CI on GitHub Actions (CPU, `--debug`).
- README contains a one-command reproduction (`make all`), expected runtimes, hardware used, the results table, and a key figure.
- Large artifacts (checkpoints, predictions) are git-ignored, with a script to regenerate them. Optionally upload final checkpoints to a release.

---

## 20. Definition of Done
- [ ] Main table: baselines, XGB per-segment, XGB global, RF, XGB spatial, GRU-nograph, STGAT, (DCRNN-lite), Hybrid × horizons 15/30/60 × MAE/RMSE/MAPE
- [ ] Peak vs off-peak and congested vs free-flow tables
- [ ] Significance tests for the key comparisons
- [ ] Ablation table + figure
- [ ] Congestion maps (interactive + static), error maps, space-time diagram, attention viz
- [ ] Optional control experiment with disclaimer
- [ ] Phase 1 and Phase 2 IEEE reports compile; all numbers auto-generated
- [ ] Slides for both phases
- [ ] Theory notes + per-member viva prep
- [ ] CI green, README reproducible, CONTRIBUTIONS.md complete