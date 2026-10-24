# Phase 2 — Exploratory Data Analysis (M2)

**Goal:** 10 figures that describe METR-LA and, most importantly, show evidence that congestion **propagates between neighboring sensors** — the motivation for the graph model.
**SPEC refs:** §6, §2, §5.2, §14 M2.
**Prerequisites:** Phase 1 (loaders, mask, split, graph).
**Owner:** Kanchan (analysis and figures)

---

## 2.1 Rules for EDA

- EDA may look at the **whole** series for description, **except** anything that informs a modeling choice (feature selection, thresholds, hyperparameters). Those use train only. Label each figure "all data" or "train only".
- All statistics computed with the missing mask (zeros excluded).
- Every figure: PNG (150 dpi) + PDF in `results/figures/eda/`, axis labels with units, a title, and colorblind-safe colors.
- Plotting code lives in `src/traffic/viz/eda_plots.py` (one function per figure). `scripts/run_eda.py` only calls them and writes stats JSON.
- Numbers quoted in `docs/eda_summary.md` are read from `results/metrics/eda_stats.json`, never typed by hand.

## 2.2 Figures (one function each)

| # | Function | Figure | Notes |
|---|---|---|---|
| 1 | `plot_missingness` | (a) sensor × day heatmap of missing rate; (b) per-sensor missing-rate histogram | Also flag days with whole-network outages (≥90% sensors missing at a timestamp). |
| 2 | `plot_daily_profile` | Mean speed by time of day, weekday vs weekend, with IQR bands | x-axis 00:00–23:55. Shade AM/PM peak windows from config. |
| 3 | `plot_tod_dow_heatmap` | Mean speed, rows = day of week, cols = time of day | 7 × 288 grid. |
| 4 | `plot_speed_distribution` | Histogram/KDE overall and peak vs off-peak | Mark the 40 mph congestion threshold. |
| 5 | `plot_acf_pacf` | ACF/PACF for 3–4 representative sensors (low/median/high variance) | Lags up to 2 × 288 to show the daily peak at 288. Use `statsmodels`; interpolate missing values for ACF only and say so. |
| 6 | `plot_corr_vs_hops` | Pairwise Pearson correlation vs shortest-path hops (box per hop count) | Use bidirectional graph for hops; unreachable pairs as a separate box. Train only. |
| 7 | `plot_lagged_xcorr` | Cross-correlation vs lag (−12…+12 steps) for 4–6 adjacent pairs on the same freeway | **Key figure.** Use first differences or detrended speed (subtract tod mean) so the daily cycle does not dominate. Mark the peak lag. Choose pairs via the graph + locations, not by eye. |
| 8 | `sensor_map` (folium) | Sensors colored by mean speed; second layer by missing rate | Save to `results/maps/eda_sensors.html`. Tooltip: sensor id, mean speed, missing %. |
| 9 | `congestion_frequency_map` | Share of weekday-peak timestamps with speed < 40 mph per sensor | folium HTML + static matplotlib scatter on lat/lon. |
| 10 | `plot_split_timeline` | Timeline of train/val/test with daily mean speed; table of weekday share, peak share, missing rate per split | Note any distribution shift (e.g. Memorial Day in which split). |

## 2.3 Propagation evidence (figure 7) — details

1. Pick candidate pairs `(i, j)` with `W[i, j] > 0` and both sensors having < 10% missing.
2. Detrend: `z = speed − tod_mean[slot, sensor]` (train tod means).
3. Compute `corr(z_i(t), z_j(t + lag))` for `lag ∈ [−12, 12]` using only timestamps where both are observed.
4. Plot, mark `argmax`. Report the distribution of peak lags across **all** adjacent pairs as a histogram (figure 7b), not only the hand-picked ones — this avoids cherry-picking.
5. Interpret: a peak at non-zero lag in the expected direction supports propagation. If peaks sit at lag 0, say so honestly.

Save the per-pair peak lags to `results/metrics/xcorr_peak_lags.csv`.

## 2.4 `docs/eda_summary.md`

For each figure: embed it, then 2–3 sentences of interpretation. End with a short "Implications for modeling" list, e.g.:
- missing-data handling choices (masked metrics, imputation default),
- why H = 12 is reasonable and why the daily lag is an optional extra,
- why peaks and long horizons are where the graph should help,
- any split distribution shift to keep in mind.

## 2.5 Notebook

`notebooks/01_eda.ipynb` imports from `traffic.viz.eda_plots` and displays the figures. No logic lives only in the notebook.

---

## Acceptance (M2)

- [ ] `make eda` regenerates all 10 figures + `eda_stats.json` from scratch
- [ ] `docs/eda_summary.md` written; every number in it traceable to `eda_stats.json`
- [ ] Propagation evidence (fig 7 + 7b) present with honest interpretation
- [ ] `make test` and `make lint` still pass

**Commit:** `M2: EDA figures and summary`
**STOP for review.**
