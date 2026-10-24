# Definition of Done (SPEC section 20): status

Legend: DONE, PARTIAL (what is missing), NOT DONE.

| Item | Status | Evidence / notes |
|---|---|---|
| Main table: baselines, XGB per-segment, XGB global, RF, XGB spatial, GRU-nograph, STGAT, Hybrid x 15/30/60 x MAE/RMSE/MAPE | DONE (DCRNN-lite not built; it was optional) | `results/metrics/main_table.csv`, README table. Hybrid = residual, gating, average. |
| Peak vs off-peak and congested vs free-flow tables | DONE | `peak_offpeak.csv`, `congested_freeflow.csv` |
| Significance tests for key comparisons | DONE | `significance.csv` (DM, HAC), `bootstrap_ci.csv`, `gap_analysis.csv` |
| Ablation table + figure | PARTIAL | A1-A12 run with **1 seed and 12 epochs** (spec: 3 seeds). A13 is read from the main table. XGB history-window sweep (A11 for ML) not run. `ablations.csv`, `results/figures/ablations/` |
| Congestion maps (interactive + static), error maps, space-time diagram, attention viz | DONE | `results/maps/*.html`, `results/figures/maps/` |
| Optional control experiment with disclaimer | DONE (CTM only; SUMO alternative not built) | `docs/control_results.md`, `docs/control_assumptions.md`. Demand is synthetic; forecasts gave no measurable benefit. |
| Phase 1 and Phase 2 IEEE reports compile; numbers auto-generated | DONE | `report/phase{1,2}/main.pdf` built with tectonic; numbers via `report/tables/numbers.tex`. Configuration constants (trial counts, row shares) are typed in prose. Authors: the five team members. |
| Slides for both phases | PARTIAL | `slides/phase1.md`, `slides/phase2.md` are Marp sources; not rendered/rehearsed. |
| Theory notes + per-member viva prep | DONE | `docs/theory/` (11 files), `docs/viva/` (5 members + shared). One viva file per member. |
| CI green, README reproducible, CONTRIBUTIONS complete | PARTIAL | Local `make lint` and `make test` pass. CI workflow written but never run (no remote). CONTRIBUTIONS lists the five members; commit history so far is from one account. |

## Deviations from the spec (all in `docs/decisions.md`)
- Reduced compute budgets: Optuna 15/6 trials, pooled XGB on 25% rows, RF on 10% rows (ADR-006).
- STGAT applies graph attention on the last 4 of 12 steps and trains on a random third of windows per epoch, max 40 epochs (ADR-007).
- DM test uses a Bartlett kernel with L = max(h-1, T^(1/3)) (ADR-005).
- DCRNN-protocol (literature-comparable) numbers were not run; only the sample-split sizes are tested.
- Control demand is synthetic (METR-LA has speed only).

## Results that do not match the spec's hypothesis (reported as found)
- Spatial features help but not more at longer horizons (gap analysis).
- STGAT equals the no-graph GRU at 15 minutes and beats it only at 60 minutes.
- The graph is not shown to beat the identity graph in the reduced-budget ablations.
- Attention shares are the same at the AM peak and at night.
- Forecast-informed ramp metering did not improve total time spent.
