# Phase 9 — Congestion Maps & Interpretability (M9)

**Goal:** maps and figures that make the results visible: forecasts on a map, where the graph helps, shockwaves in a space-time diagram, and GAT attention.
**SPEC refs:** §13, §9.3, §14 M9.
**Prerequisites:** Phases 4, 6 (Phase 7 optional for hybrid maps).
**Owner:** Vaibhav (maps, space-time, summary plots); Pranav Singh (attention analysis)

---

## 9.1 Color scheme (shared constant in `viz/__init__.py`)

| Speed | Color |
|---|---|
| < 40 mph | red (congested) |
| 40–55 mph | orange |
| > 55 mph | green |
| missing | grey, hollow marker |

## 9.2 Interactive folium maps — `src/traffic/viz/congestion_map.py`

```python
def forecast_map(pred_files: dict[str, Path], anchor_ts, horizons=(3, 6, 12), out: Path) -> None
def animated_peak_map(pred_file, start_ts, end_ts, horizon, out: Path) -> None  # TimestampedGeoJson
```
- Pick the AM-peak anchor **programmatically**: the test weekday timestamp (07:00–09:00) with the largest number of congested sensors. Save the choice to `results/maps/selected_timestamps.json` so it is reproducible.
- Layers: (a) ground truth, (b) STGAT, (c) XGB-global, each at +15/+30/+60 min; layer control to toggle.
- Tooltip: sensor id, true speed, predicted speed, abs error.
- Animated map: one AM peak hour (12 frames), truth vs STGAT side by side (two maps or two layers).
- Output: `results/maps/forecast_{ts}.html`, `results/maps/animated_peak.html`.

## 9.3 Error maps

```python
def error_diff_map(per_sensor_csv, model_a="xgb_global", model_b="stgat", horizon=12, out=...) -> None
```
- Per-sensor `MAE_A − MAE_B` with a **diverging** colormap centered at 0 (blue = graph better, red = XGB better).
- Do this for h = 3 and h = 12 and for peak-only. Add a short note: are graph gains clustered at freeway interchanges / bottlenecks?

## 9.4 Static maps (report)

matplotlib scatter on lat/lon (equal aspect after `cos(lat)` correction); optional `contextily` basemap behind a try/except so it never breaks the build when offline. PDF + PNG to `results/figures/maps/`.

## 9.5 Space-time diagram — `src/traffic/viz/horizon_plots.py::space_time_diagram`

1. Choose one freeway corridor: a chain of ≥ 10 sensors ordered along the road. Derive order from the directed graph (longest path in the `forward` adjacency restricted to one freeway's sensors, or sort by projected position along the corridor's principal axis). Save the ordered sensor list to `results/metrics/corridor.json` — Phase 10 reuses it.
2. Plot sensors (y, in travel order) × time (x, one test weekday 05:00–11:00), colored by speed.
3. Three panels: truth, STGAT +30 min forecast, XGB-global +30 min forecast (forecast placed at its target time).
4. Annotate a backward-moving congestion front (shockwave) if visible — describe, don't overclaim.

## 9.6 Attention visualization — `src/traffic/viz/attention_viz.py`

- Load the best STGAT seed; run on selected test windows with `return_attn=True`.
- For 3–4 sensors near bottlenecks: draw arrows/lines from each neighbor j to the target i, width ∝ α_ij (average over heads; also a per-head small-multiples figure).
- Compare **AM peak** vs **night (02:00–04:00)**, averaged over all test weekdays at those times.
- Quantify: share of attention mass on upstream vs downstream vs self, peak vs night → `results/metrics/attention_summary.csv`.
- Caption must say: attention is a qualitative observation, **not** causal evidence.

## 9.7 Summary figures — `src/traffic/viz/horizon_plots.py`

- **Key figure:** MAE vs horizon (1..12) for all models on one plot (ML models with only {3, 6, 12} as markers). DL with ±std band.
- Peak vs off-peak grouped bars at 15/30/60.
- Ablation bars already exist from Phase 8.

`scripts/make_maps.py` runs everything in this phase.

---

## Acceptance (M9)

- [ ] Interactive maps (truth / STGAT / XGB at 3 horizons) + animated peak map open in a browser
- [ ] Error-difference maps (h = 3, 12, peak)
- [ ] Space-time diagram with corridor saved
- [ ] Attention figures + `attention_summary.csv`
- [ ] Key error-vs-horizon figure for all models

**Commit:** `M9: congestion maps, space-time diagram, attention viz`
**STOP for review.**
