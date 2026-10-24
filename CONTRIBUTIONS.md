# Contributions

Work is split in **decreasing order of difficulty and technical depth**: Pranav Singh owns the hardest, most central parts; Vaibhav the lightest. Everyone must still be able to explain the shared topics (split protocol, masking, adjacency, GAT equations, hybrid OOF logic, headline results) in the individual viva; see `docs/viva/`.

| Rank | Member | Role |
|---|---|---|
| 1 | **Pranav Singh** | Technical lead: deep-learning core and hybrid models |
| 2 | **Kunal** | Phase 1 machine learning |
| 3 | **Himanshu** | Evaluation and statistics framework, simulated control |
| 4 | **Kanchan** | Data pipeline, graph, missing data, EDA |
| 5 | **Vaibhav** | Scaffold, visualisation, documentation, report assembly |

## 1. Pranav Singh (hardest and most technical)
- **Modules:** `models/layers/gat_dense.py` (dense + sparse GAT, equivalence with PyG), `models/stgat.py`, `models/gru_nograph.py`, `models/hybrid.py`, `models/factory.py`, `training/trainer.py`, `training/losses.py`, `data/datasets.py` (fast batch iterator), `viz/attention_viz.py`, `scripts/train_dl.py`, `scripts/make_oof.py`, `scripts/train_hybrid.py`, `scripts/run_ablations.py`, `scripts/run_dl_chain.sh`, `scripts/run_ablation_chain.sh`.
- **Tests:** `test_gat_layer.py`, `test_stgat_shapes.py`, `test_hybrid.py`.
- **Experiments:** STGAT and GRU-nograph (3 seeds), residual / gating / average hybrids with leakage-safe out-of-fold forecasts, ablations A1-A12, the compute-driven STGAT simplification (ADR-007), architecture and hybrid ADRs (ADR-007, 008).
- **Report sections:** Phase 2 IV (Architecture), V (Hybrid Method), VIII (Ablations), plus review of both abstracts and conclusions.
- **Theory notes:** `gat.md`, `hybrid.md`, `spatiotemporal_modeling.md`, `diffusion_conv.md`.
- **Also:** final integration and review of every pull request.

## 2. Kunal
- **Modules:** `features/lags.py`, `features/spatial.py`, `features/build_tabular.py`, `models/ml_models.py`, `training/tuning.py`, `scripts/train_ml.py`, `scripts/ml_analysis.py`, `scripts/run_ml_chain.sh`, `configs/ml/*`.
- **Tests:** `test_features_no_future.py`, `test_tabular_alignment.py`.
- **Experiments:** XGBoost per-segment / pooled / spatial, Random Forest, Optuna tuning, SHAP and gain importance, gap analysis, the daily-lag side experiment, ADR-006 (reduced ML budget).
- **Report sections:** Phase 1 V (Feature Engineering), VI (Methods), IX results on SHAP and gap analysis; Phase 2 section on ML baselines in the setup.
- **Theory notes:** `xgboost.md`, `random_forest.md`.

## 3. Himanshu
- **Modules:** `evaluation/metrics.py`, `evaluation/slices.py`, `evaluation/stats_tests.py`, `evaluation/io.py`, `evaluation/report_tables.py`, `evaluation/report_numbers.py`, `models/baselines.py`, `control/*` (CTM, ALINEA, predictive ALINEA, experiment), `scripts/run_baselines.py`, `scripts/evaluate_all.py`, `scripts/evaluate_ablations.py`, `scripts/hybrid_analysis.py`, `scripts/run_control.py`, `scripts/make_report_assets.py`.
- **Tests:** `test_metrics_masking.py`, `test_baselines.py`, `test_ctm.py`.
- **Experiments:** baselines, all metric tables, Diebold-Mariano and bootstrap results, hybrid gain decomposition, the simulated ramp-metering study (disclaimer in every output), ADR-005.
- **Report sections:** Phase 1 VII (Experimental Setup: metrics, significance), VIII tables; Phase 2 VI (Experimental Setup), VII (Results tables, significance), X (Simulated Control).
- **Theory notes:** `metrics.md`, `significance_testing.md`, `control.md`.

## 4. Kanchan
- **Modules:** `data/load.py`, `data/graph.py`, `data/missing.py`, `data/split.py`, `data/scaling.py`, `data/windows.py`, `data/pipeline.py`, `features/temporal.py`, `viz/eda_plots.py`, `scripts/download_data.py`, `scripts/prepare_data.py`, `scripts/run_eda.py`, `scripts/make_eda_summary.py`.
- **Tests:** `test_load.py`, `test_graph.py`, `test_split_no_leakage.py`, `test_scaling_train_only.py`, `test_windows.py`.
- **Experiments:** data download and integrity checks, graph rebuild and statistics, missing-data analysis, the ten EDA figures and the propagation evidence, ADR-002, 003, 004.
- **Report sections:** Phase 1 III (Dataset and EDA), IV (Problem Formulation); Phase 2 III (Data, Graph and Regularisation).
- **Theory notes:** `chronological_splits.md`, `missing_data.md`.

## 5. Vaibhav (lightest)
- **Modules:** repository scaffold (`pyproject.toml`, `Makefile`, `.github/workflows/ci.yml`, `.gitignore`, `configs/base.yaml` upkeep), `viz/congestion_map.py`, `viz/horizon_plots.py`, `scripts/make_maps.py`, `scripts/make_docs.py`, `README.md`, `slides/phase1.md`, `slides/phase2.md`, `report/refs.bib` (verifying every entry), LaTeX assembly of both reports.
- **Tests:** keeps `make lint` and `make test` green (CI), `test_config.py`, `test_end_to_end_debug.py`.
- **Experiments:** runs the full pipeline end to end with the `make` targets, produces the interactive maps, error maps and space-time diagram, keeps `docs/experiment_log.md` current.
- **Report sections:** Phase 1 I (Introduction), II (Related Work), X (Conclusion); Phase 2 I, II (Literature), IX (Interpretability and Maps), XI (Limitations and Ethics), XII (Conclusion); references; final PDF builds.
- **Docs:** `docs/definition_of_done.md`, `docs/viva/` index, per-member file format.

## Difficulty-order rationale
1. The GAT layer, STGAT, out-of-fold hybrid and trainer carry the project's core technical risk (numerical equivalence with PyG, leakage-safe stacking, GPU training under compute limits).
2. Feature engineering and tuned tree models are broad but well-understood, with real leakage pitfalls.
3. The evaluation and statistics code (HAC Diebold-Mariano, block bootstrap) and the control simulator are small but easy to get subtly wrong.
4. The data pipeline and EDA are crucial but follow a clear recipe.
5. Visualisation, documentation and report assembly are the most mechanical.

## Git hygiene
Commits should come from each member's own account on their owned modules (pull requests reviewed by Pranav). The initial history was created from one account while the project was bootstrapped; going forward use feature branches named `<member>/<topic>`.
