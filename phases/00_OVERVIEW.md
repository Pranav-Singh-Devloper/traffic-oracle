# Build Plan — Overview

This folder splits [SPEC.md](../SPEC.md) into 12 build phases, one per milestone in SPEC §14.
Build them **in order**. Each phase ends with tests, acceptance checks, a commit, a short summary, and a **stop for human review**.

| Phase | File | Milestone | Main output | Owner |
|---|---|---|---|---|
| 0 | [PHASE_00_SCAFFOLD.md](PHASE_00_SCAFFOLD.md) | M0 | Repo skeleton, configs, CI, empty tests | Vaibhav |
| 1 | [PHASE_01_DATA_AND_GRAPH.md](PHASE_01_DATA_AND_GRAPH.md) | M1 | Download, loaders, graph, mask, split, scaler, windows | Kanchan |
| 2 | [PHASE_02_EDA.md](PHASE_02_EDA.md) | M2 | 10 EDA figures + `docs/eda_summary.md` | Kanchan |
| 3 | [PHASE_03_BASELINES_AND_EVALUATION.md](PHASE_03_BASELINES_AND_EVALUATION.md) | M3 | Baselines, metrics, slices, significance, tables | Himanshu |
| 4 | [PHASE_04_ML_MODELS.md](PHASE_04_ML_MODELS.md) | M4 | XGB per-segment / global / spatial, RF, SHAP, gap analysis | Kunal |
| 5 | [PHASE_05_PHASE1_REPORT.md](PHASE_05_PHASE1_REPORT.md) | M5 | Phase 1 IEEE report + slides | Vaibhav (+ all) |
| 6 | [PHASE_06_DL_MODELS.md](PHASE_06_DL_MODELS.md) | M6 | Dense GAT, GRU-nograph, STGAT, trainer, (DCRNN-lite) | Pranav Singh |
| 7 | [PHASE_07_HYBRID.md](PHASE_07_HYBRID.md) | M7 | OOF XGB, residual hybrid, gating, average | Kunal, Himanshu |
| 8 | [PHASE_08_ABLATIONS.md](PHASE_08_ABLATIONS.md) | M8 | A1–A13 table + figure | Kanchan, Himanshu |
| 9 | [PHASE_09_MAPS_AND_INTERPRETABILITY.md](PHASE_09_MAPS_AND_INTERPRETABILITY.md) | M9 | Folium maps, error maps, space-time diagram, attention viz | Vaibhav |
| 10 | [PHASE_10_SIMULATED_CONTROL.md](PHASE_10_SIMULATED_CONTROL.md) | M10 (optional) | CTM + ALINEA + predictive ALINEA | Himanshu |
| 11 | [PHASE_11_PHASE2_REPORT_AND_VIVA.md](PHASE_11_PHASE2_REPORT_AND_VIVA.md) | M11 | Phase 2 report, slides, theory notes, viva prep | Vaibhav (+ all) |

**Team T042 (ordered by difficulty):** Pranav Singh, Kunal, Himanshu, Kanchan, Vaibhav; see `../CONTRIBUTIONS.md`.

## Dependency graph

```
P0 → P1 → P2
        ↘ P3 → P4 → P5
                 ↘ P6 → P7 → P8 → P9 → P10 → P11
```
P2 (EDA) can run in parallel with P3. P10 needs a trained STGAT (P6). P11 needs everything.

## Rules that apply to every phase (from SPEC header)

1. **No fabricated numbers.** Every number in README, report, slides comes from a script writing to `results/`. If a run fails, report the failure.
2. **No future leakage.** Chronological splits. Scalers, imputers, encoders, target encoders, historical averages: fit on **train timestamps only**.
3. **Control is simulation only.** Never claim live/real-world control.
4. **Readable code.** Type hints, NumPy-style docstrings that say *why*. Every member must be able to explain every file.
5. **CPU `--debug` + GPU full.** Auto-detect device (`cuda` → `mps` → `cpu`).
6. **Deviations go in `docs/decisions.md`** (ADR format) *before* deviating.
7. **`speed == 0` is missing**, never real zero speed. Mask it in losses and metrics.

## Dataset source (DCRNN repository)

From <https://github.com/liyaguang/DCRNN#data-preparation>:

| File | Where it comes from |
|---|---|
| `metr-la.h5` | Google Drive folder `https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX` (also Baidu Yun `https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g`). Not in git. Same folder holds `pems-bay.h5` (PeMS-BAY extension). |
| `adj_mx.pkl` | `https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl` |
| `distances_la_2012.csv` | `.../data/sensor_graph/distances_la_2012.csv` |
| `graph_sensor_ids.txt` | `.../data/sensor_graph/graph_sensor_ids.txt` |
| `graph_sensor_locations.csv` | `.../data/sensor_graph/graph_sensor_locations.csv` |

DCRNN reference commands (for the *literature-comparable* protocol only — we do not depend on the DCRNN code):
```bash
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5
python -m scripts.gen_adj_mx --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt \
    --normalized_k=0.1 --output_pkl_filename=data/sensor_graph/adj_mx.pkl
```

## Repository root

SPEC §4 names the repo `traffic-gnn-forecasting/`. This folder (`DL_AML_Project/`) **is** that root.
Note: `DL_AML_Project/` currently sits inside a parent git repo (`folder_2/`). Phase 0 decides whether to `git init` a standalone repo here (recommended — GitHub Actions CI and per-member commit history need their own repo). Record the choice as ADR-001.

## End-of-phase checklist (copy into every phase summary)

- [ ] `make test` passes
- [ ] Phase acceptance checks pass (listed in each file)
- [ ] `ruff check . && black --check .` clean
- [ ] New decisions recorded in `docs/decisions.md`
- [ ] Run logged in `docs/experiment_log.md` (config + git hash)
- [ ] Commit with message `M<k>: <summary>`
- [ ] Print summary: what was built, what numbers came out (read from `results/`), what failed, open questions
- [ ] **STOP for human review**
