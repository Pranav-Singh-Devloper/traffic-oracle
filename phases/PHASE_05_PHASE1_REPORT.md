# Phase 5 — Phase 1 Report & Slides (M5)

**Goal:** an IEEE-format Phase 1 report and a slide outline. Every number is generated, never typed.
**SPEC refs:** §17 (Phase 1 structure), §18 (literature), §15 (theory topics 1–3, 8, 9), §14 M5.
**Prerequisites:** Phase 4.

---

## 5.1 Asset generation — `scripts/make_report_assets.py`

- Copies figures from `results/figures/` → `report/figures/` (PDF preferred).
- Writes LaTeX tables to `report/tables/` (`booktabs`, via `report_tables.py`):
  - `main_table_phase1.tex` (baselines + ML at 15/30/60, MAE/RMSE/MAPE)
  - `peak_offpeak_phase1.tex`, `congested_freeflow_phase1.tex`
  - `gap_analysis.tex`, `significance_phase1.tex`
  - `dataset_stats.tex` (from `eda_stats.json`, `graph_stats.json`, `split_boundaries.json`)
- Writes `report/tables/numbers.tex` with `\newcommand` macros for every number quoted in prose, e.g. `\newcommand{\xgbGlobalMAEsixty}{<value read from main_table.csv>}`. **Prose uses only these macros.** This is how "every number comes from a script" is enforced.
- Add a check: grep `main.tex` for digits followed by "mph" or "%" outside macros and warn.

## 5.2 `report/phase1/main.tex` (IEEEtran, conference)

Sections (SPEC §17):
1. **Abstract** — problem, data, models, headline result via macros.
2. **Introduction** — why forecasting matters; the propagation hypothesis; contributions list.
3. **Related Work** — classical (historical average, ARIMA), ML (tree ensembles), early graph models (DCRNN, STGCN). Cite from `refs.bib`.
4. **Dataset & EDA** — METR-LA facts (speed only, zero = missing, freeway loop detectors), missingness, daily profile, **propagation evidence** (fig 7).
5. **Problem Formulation** — `X ∈ R^{H×N×C} → Y ∈ R^{F×N}`, masked loss, notation table.
6. **Feature Engineering** — lags, rolling stats, cyclic encoding (why), spatial features.
7. **Methods** — baselines; XGBoost (second-order objective, gain, λ/γ); Random Forest (bagging, feature subsampling, variance reduction, multi-output splits); XGB-spatial.
8. **Experimental Setup** — strict chronological split (and why random splits leak), train-only scaling/imputation, Optuna budget, metrics + MAPE threshold, DM + bootstrap.
9. **Results** — main table, error vs horizon, peak/off-peak, SHAP, gap analysis.
10. **Discussion & Limitations** — single city/season, speed only, fixed distance graph, MAPE instability.
11. **Conclusion & Phase 2 Plan** — STGAT, GRU-nograph ablation, hybrid.

Keep to the course page limit (ask the team; default 6–8 pages).

## 5.3 `report/refs.bib`

Add every entry from SPEC §18. **Verify each entry** (authors, venue, year, pages) against the paper or DBLP before finalizing. Do not invent DOIs; omit the field if unsure. Mark verified entries with a `% verified: <source>` comment.

## 5.4 Theory notes needed now (`docs/theory/`)

Write these four now (rest in Phase 11), each with equations, intuition, and 5+ likely viva Q&A:
- `chronological_splits.md`, `xgboost.md`, `random_forest.md`, `metrics.md`

## 5.5 Slides — `slides/phase1.md`

~12 slides, Marp-compatible Markdown:
title → problem & hypothesis → data facts → missingness → propagation evidence → features → models → setup (split/leakage) → main results → gap analysis → SHAP → limitations & Phase 2 plan.
Each slide: ≤ 5 bullets + one figure path. Speaker notes as HTML comments.

---

## Acceptance (M5)

- [ ] `cd report/phase1 && latexmk -pdf main.tex` compiles without errors
- [ ] Every number in prose comes from `numbers.tex` macros; every table from `report/tables/`
- [ ] All bib entries marked verified (or flagged for the team)
- [ ] `slides/phase1.md` complete
- [ ] Four theory notes written

**Commit:** `M5: Phase 1 report, slides, theory notes`
**STOP for review.**
