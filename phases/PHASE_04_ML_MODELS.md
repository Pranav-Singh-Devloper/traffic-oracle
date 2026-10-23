# Phase 4 — Phase 1 Machine Learning (M4)

**Goal:** tabular features, four ML models, tuning, SHAP, and the **gap analysis** (does a hand-made spatial feature help, and where?).
**SPEC refs:** §7.2–7.5, §8.3, §14 M4.
**Prerequisites:** Phases 1 and 3.

---

## 4.1 Tabular features — `src/traffic/features/`

One row per (sensor i, anchor time t). Build with vectorized numpy over `[T, N]` arrays, then reshape to rows. **Everything at anchor t uses data at times ≤ t only.**

### `lags.py`
```python
def lag_features(speed_in, mask, H=12) -> dict[str, np.ndarray]   # each [T, N]
```
- `lag_0 … lag_11` = imputed input speed at t, t−1, …, t−11
- rolling mean/std/min/max over 3, 6, 12 steps (on imputed input)
- `diff_1 = lag_0 − lag_1`, `diff_3 = lag_0 − lag_3`
- `miss_0 … miss_11` (is_missing flags) and `n_missing_12`
- `lag_288` only when `use_daily_lag: true` (default **off**)

### `temporal.py` (from Phase 1)
Cyclic tod/dow, `is_weekend`, `is_holiday` — taken at the **anchor** time t.

### `spatial.py` (only for `xgb_spatial`)
```python
def neighbor_features(speed_in, W, lags=(0, 1, 2)) -> dict[str, np.ndarray]
```
For sensor i, row-normalized weights:
- `up_mean_lag{k}` — weighted mean over upstream neighbors (from `Wᵀ`, per ADR-002 convention)
- `down_mean_lag{k}` — weighted mean over downstream neighbors (from `W`)
- `hop2_mean_lag0` — mean over 2-hop neighbors (`A²` binarized, excluding 1-hop and self)
- `own_minus_up`, `own_minus_down` at lag 0
- Sensors with no neighbors in a direction → NaN (XGBoost handles NaN natively; RF needs fill — document).

### Static features (global models)
`sensor_idx` (categorical for XGBoost with `enable_categorical=True`), `lat`, `lon`, `in_degree`, `out_degree`. If target encoding is used, fit on **train** only.

### `build_tabular.py`
```python
def build_design_matrix(cfg, split: str, model_kind: str, horizon: int | None) -> (X: pd.DataFrame, y, row_meta)
```
- Targets: `y = speed[t + h, i]`; drop rows where target is missing (equivalent to masking, SPEC §7.4).
- Anchors per split follow the strict protocol, i.e. **the same anchor set as the DL windows** so test rows match the common prediction format exactly.
- `row_meta` keeps `(anchor_idx, sensor_idx)` to scatter predictions back into `[S, F, N]`.
- Memory: full global train ≈ 24k anchors × 207 sensors ≈ 5M rows × ~60 cols float32 ≈ 1.2 GB. Use float32 and build per horizon. Allow a `row_subsample` config for RF; record it in an ADR.
- PeMS hook: accept extra channels (`flow`, `occupancy`) and generate their lags if present; no-op for METR-LA.

---

## 4.2 Models — `src/traffic/models/ml_models.py`

| Model | Config | Training unit | Notes |
|---|---|---|---|
| `xgb_per_segment` | `configs/ml/xgb_per_segment.yaml` | 1 model per sensor × horizon {3, 6, 12} = 621 | `tree_method="hist"`, `n_jobs=1` per model, joblib parallel over sensors. Flag `all_horizons: true` → 2,484 models. No static features. |
| `xgb_global` | `configs/ml/xgb_global.yaml` | 1 pooled model per horizon (12) | Includes static sensor features. No neighbor features. |
| `rf_multioutput` | `configs/ml/rf_multioutput.yaml` | 1 pooled `RandomForestRegressor` predicting all 12 horizons jointly | Rows with *any* missing target: either drop, or fill target with persistence for training only and document. Recommend dropping and logging the share dropped. Fill NaN neighbor features. |
| `xgb_spatial` | `configs/ml/xgb_spatial.yaml` | Same as `xgb_global` + neighbor features | The bridge experiment. |

Common interface:
```python
class MLForecaster(Protocol):
    def fit(self, X_train, y_train, X_val, y_val) -> None
    def predict(self, X) -> np.ndarray
    def save(self, path) / load(path)
```
- XGBoost early stopping on val (`early_stopping_rounds=50`, eval metric MAE).
- Final models are trained on **train only** (not train+val) to match DL. Write the ADR.
- `scripts/train_ml.py --model {xgb_per_segment|xgb_global|rf|xgb_spatial} [--debug] [--tune]` → writes `results/predictions/{model}.npz` (NaN for horizons not trained) and saves models under `checkpoints/ml/` (git-ignored).

---

## 4.3 Tuning — `src/traffic/training/tuning.py`

- Optuna TPE sampler, seeded. Objective = masked val MAE averaged over horizons {3, 6, 12}.
- Budget: 40 trials per model (per-segment: tune on a representative sample of ~20 sensors with shared hyperparameters, then apply to all — document).
- Search space (XGB): `max_depth 3–10`, `learning_rate 0.01–0.3 log`, `n_estimators` via early stopping (cap 2000), `subsample 0.5–1`, `colsample_bytree 0.5–1`, `min_child_weight 1–20`, `reg_lambda 1e-3–10 log`, `gamma 0–5`.
- RF: `n_estimators 100–500`, `max_depth 8–30|None`, `max_features {sqrt, 0.3, 0.5}`, `min_samples_leaf 1–20`.
- Log every trial to `results/metrics/tuning_{model}.csv`; best params to `configs/ml/{model}.best.yaml`.
- **Never** touch the test split during tuning. Add an assertion that the tuning code receives no test indices.

---

## 4.4 Analysis

### Feature importance + SHAP — `scripts/ml_analysis.py` (add to tree)
- XGBoost gain importance (top 25) for `xgb_global` and `xgb_spatial`, h ∈ {3, 12}.
- SHAP `TreeExplainer` on a **random sample of 5,000 test rows** (seeded): beeswarm + mean |SHAP| bar.
- Question to answer in text: which lags/time features matter, and do neighbor features rank high in `xgb_spatial`, especially at h = 12?

### Error curves
- MAE vs horizon (1..12 where available) for all ML models + baselines → `results/figures/ml/error_vs_horizon.png`.
- Per-sensor MAE map for `xgb_global` (folium + static) → which locations are hardest?

### Gap analysis (the core argument)
- Table + figure: `ΔMAE = MAE(xgb_global) − MAE(xgb_spatial)` by horizon × {peak, off-peak, weekend} × {congested, free-flow}, with block-bootstrap CIs and DM tests.
- Save to `results/metrics/gap_analysis.csv`, figure `results/figures/ml/gap_analysis.png`.
- Write `docs/gap_analysis.md`: if the gain grows with horizon and at peaks → supports the propagation hypothesis. If it doesn't, say so plainly.

### Daily-lag side experiment
Re-run `xgb_global` with `use_daily_lag: true`; report separately (not in the main equal-window table).

---

## 4.5 Tests

- `test_features_no_future.py` (add): perturb `speed[t+1:]` → features at anchor t unchanged.
- `test_tabular_alignment.py` (add): scattered predictions from `row_meta` land in the right `[S, F, N]` cells (round-trip with a dummy model that returns the target).
- End-to-end debug: every ML model trains + predicts in `--debug` mode in < 60 s on CPU.

---

## Acceptance (M4)

- [ ] All 4 ML models in `main_table.csv` with peak/off-peak slices
- [ ] Tuning logs exist for each model
- [ ] SHAP figures + gap analysis figure/CSV/markdown exist
- [ ] Every ML model beats persistence at 15 min (or the failure is reported and explained)
- [ ] Sanity warnings reviewed

**Commit:** `M4: Phase 1 ML models, tuning, SHAP, gap analysis`
**STOP for review.**
