# Phase 3 — Baselines + Evaluation Framework (M3)

**Goal:** one evaluation pipeline that every model goes through, plus three naive baselines that every model must beat.
**SPEC refs:** §7.1, §8 (all), §14 M3.
**Prerequisites:** Phase 1.

---

## 3.1 Common prediction format

Every model writes `results/predictions/{model_name}[_seed{s}].npz`:

| Key | Shape | Meaning |
|---|---|---|
| `y_pred` | `[S, F, N]` float32 | predictions in **mph** |
| `y_true` | `[S, F, N]` float32 | ground truth in mph (0 where missing) |
| `mask` | `[S, F, N]` bool | `y_true > 0` |
| `target_ts` | `[S, F]` int64 (ns) | target timestamps |
| `sensor_ids` | `[N]` str | pkl order |
| `meta` | json string | model name, config hash, git hash, protocol, H |

`src/traffic/evaluation/io.py` (add to tree): `save_predictions(...)`, `load_predictions(...)`, and `assert_aligned(a, b)` which checks that two prediction files share the same `target_ts`, `mask`, and `sensor_ids` (fairness rule §8.3). ML models that predict only horizons {3, 6, 12} store `NaN` for the others and the evaluator skips NaN horizons.

---

## 3.2 Baselines — `src/traffic/models/baselines.py`

```python
def persistence(windows) -> np.ndarray            # ŷ(t+h) = last observed input value y(t)
def historical_average(train_speed, train_mask, train_index, target_ts) -> np.ndarray
    # mean of that sensor at the same (dow, tod) slot, train only; fallback chain: (tod) → sensor mean
def seasonal_naive(full_speed, target_ts, period=288) -> np.ndarray
    # ŷ(t+h) = y(t+h−288)
```
Details:
- **Persistence** with missing last value: use the last *observed* value within the input window (imputed input is fine since it only uses the past). Document.
- **Seasonal naive** looks back 288 steps — outside the H = 12 window. That is allowed for a baseline but label it. If `y(t+h−288)` is missing, fall back to the historical average and log the fallback rate. For the first test day, the lookup lands in val data — still the past, so no leakage.
- `scripts/run_baselines.py` (add to tree) writes the three `.npz` files.

---

## 3.3 Metrics — `src/traffic/evaluation/metrics.py`

```python
def masked_mae(y, yhat, m) -> float     # Σ m|y−ŷ| / Σ m
def masked_rmse(y, yhat, m) -> float    # sqrt(Σ m(y−ŷ)² / Σ m)
def masked_mape(y, yhat, m, min_y=1.0) -> float   # m' = m ∧ (y ≥ min_y);  ×100
```
- All-masked input returns `np.nan` (not 0, not an error) and logs a warning.
- Ignore `NaN` predictions (horizons not produced) by AND-ing them out of the mask.
- MAPE threshold `min_y = 1.0` mph — record as ADR (SPEC §8.1 asks to document the choice).
- Torch versions for training live in `training/losses.py` (Phase 6) and must agree with these numerically — add a test then.

`tests/test_metrics_masking.py` — **hand-computed** examples:
- 2×2 example with one masked entry → check MAE/RMSE/MAPE to 1e-9.
- All masked → NaN.
- Masked entries with huge errors do not change the result.
- MAPE excludes `y < 1`.

---

## 3.4 Slices — `src/traffic/evaluation/slices.py`

```python
def slice_masks(target_ts, y_true, cfg) -> dict[str, np.ndarray]   # each [S, F] or [S, F, N] bool
```
Slices (all judged on the **target** timestamp, LA local time as stored):
- `all`
- `horizon_h` for h = 1..12 (headline 3, 6, 12 = 15/30/60 min)
- `peak_am`, `peak_pm`, `peak` (weekday only, windows from config), `offpeak_weekday`, `weekend`
- `congested` (y < 40 mph), `freeflow` (y ≥ 40)
- per sensor (returns `[N]` metrics for maps)

`evaluate(pred_file, cfg) -> pd.DataFrame` → long format: `model, seed, slice, horizon, metric, value, n_obs`.

---

## 3.5 Significance — `src/traffic/evaluation/stats_tests.py`

```python
def diebold_mariano(e1, e2, h, loss="abs") -> (dm_stat, p_value)
def block_bootstrap_ci(err, mask, block_len=288, n_boot=1000, alpha=0.05, seed=42) -> (lo, hi)
```
- Inputs for DM: per-timestamp loss series `d_t = mean_over_sensors(|e1|) − mean_over_sensors(|e2|)` for one horizon, using only sensors observed in both.
- HAC variance: Newey–West with `h − 1` lags (rectangular kernel per DM 1995), falling back to Bartlett if the variance comes out negative. Apply the Harvey–Leybourne–Newbold (1997) small-sample correction and use a t-distribution with `T − 1` df.
- Test windows overlap (stride 1), so losses are autocorrelated — that is why HAC is needed. Say so in the docstring.
- Block bootstrap: resample whole days (288-step blocks) of test timestamps with replacement; recompute masked MAE per horizon.
- Tests: DM on identical errors → stat 0, p = 1; on clearly different errors → p < 0.01; bootstrap CI contains the point estimate.

---

## 3.6 Tables — `src/traffic/evaluation/report_tables.py` + `scripts/evaluate_all.py`

`scripts/evaluate_all.py`:
1. Load every `.npz` in `results/predictions/`; `assert_aligned` against a reference (persistence).
2. Evaluate all slices → `results/metrics/all_metrics_long.csv`.
3. Aggregate seeds (mean ± std) for DL models.
4. Write:
   - `results/metrics/main_table.csv` — model × horizon {15, 30, 60} × {MAE, RMSE, MAPE}
   - `results/metrics/peak_offpeak.csv`
   - `results/metrics/congested_freeflow.csv`
   - `results/metrics/per_sensor_mae.csv`
   - `results/metrics/significance.csv` (pairs from config list; empty for now)
   - `results/metrics/bootstrap_ci.csv`
5. **Sanity warnings** (SPEC §8.6), printed and saved to `results/metrics/sanity_warnings.txt`:
   - any model MAE@60 < 3.0 → "possible leakage, investigate"
   - any learned model not beating persistence at 15 min → "something is broken"
   - any prediction file misaligned → hard error

`report_tables.py`: CSV → Markdown and LaTeX (`booktabs`, best value per column in bold, mean ± std for DL). `scripts/make_report_assets.py` writes `report/tables/*.tex` and replaces the README block between `<!-- RESULTS_START -->` and `<!-- RESULTS_END -->`.

---

## Acceptance (M3)

- [ ] `results/metrics/main_table.csv` contains persistence, historical average, seasonal naive at 15/30/60 min
- [ ] Peak/off-peak and congested/free-flow CSVs exist for baselines
- [ ] `test_metrics_masking.py` + stats tests pass
- [ ] README results block auto-filled (baselines only)
- [ ] Sanity check: persistence MAE grows with horizon; historical average is roughly flat across horizons. If not, investigate before moving on.

**Commit:** `M3: baselines and evaluation framework`
**STOP for review.**
