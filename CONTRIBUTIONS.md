# Contributions: Team T042

**Team T042**: Pranav Singh, Kunal, Himanshu, Kanchan, Vaibhav.

Work is split in **decreasing order of difficulty and technical depth**, with workload kept deliberately even: Pranav Singh owns the hardest, most central technical core; Vaibhav the most mechanical parts, but everyone owns real code, experiments, report sections and theory notes. Everyone must still be able to explain the shared topics (split protocol, masking, adjacency, GAT equations, hybrid OOF logic, headline results) in the individual viva; see `docs/viva/`.

| Rank | Member | Role |
|---|---|---|
| 1 | **Pranav Singh** | Technical lead: deep-learning core (GAT layer, STGAT, GRU, trainer) |
| 2 | **Kunal** | Phase 1 machine learning and the hybrid models |
| 3 | **Himanshu** | Evaluation and statistics framework, baselines, simulated control |
| 4 | **Kanchan** | Data pipeline, graph, missing data, EDA, ablation runs |
| 5 | **Vaibhav** | Scaffold/CI, maps and attention plots, README, slides, report assembly |

## 1. Pranav Singh (hardest and most technical)
- **Modules:** `models/layers/gat_dense.py` (dense + sparse GAT, equivalence with PyG), `models/stgat.py`, `models/gru_nograph.py`, `models/factory.py`, `training/trainer.py`, `training/losses.py`, `data/datasets.py` (fast batch iterator), `scripts/train_dl.py`, `scripts/run_dl_chain.sh`.
- **Tests:** `test_gat_layer.py`, `test_stgat_shapes.py`.
- **Experiments:** STGAT and GRU-nograph (3 seeds), the compute-driven STGAT simplification and its timing study (ADR-007), architecture decisions, final integration and review of every pull request.
- **Report sections:** Phase 2 IV (Architecture) and the graph-vs-no-graph discussion in VII (Results), XII (Conclusion); review of both abstracts.
- **Theory notes:** `gat.md`, `spatiotemporal_modeling.md`.

## 2. Kunal
- **Modules:** `features/lags.py`, `features/spatial.py`, `features/build_tabular.py`, `models/ml_models.py`, `training/tuning.py`, `models/hybrid.py`, `scripts/train_ml.py`, `scripts/run_ml_chain.sh`, `scripts/make_oof.py`, `scripts/train_hybrid.py`, `scripts/ml_analysis.py`, `configs/ml/*`, `configs/hybrid/*`.
- **Tests:** `test_features_no_future.py`, `test_tabular_alignment.py`, `test_hybrid.py`.
- **Experiments:** XGBoost per-segment / pooled / spatial, Random Forest, Optuna tuning, SHAP and gain importance, gap analysis, daily-lag experiment (ADR-006); out-of-fold forecasts, residual / gating / average hybrids (ADR-008).
- **Report sections:** Phase 1 V (Feature Engineering), VI (Methods), SHAP and gap-analysis results; Phase 2 V (Hybrid Method).
- **Theory notes:** `xgboost.md`, `hybrid.md`.

## 3. Himanshu
- **Modules:** `evaluation/metrics.py`, `evaluation/slices.py`, `evaluation/stats_tests.py`, `evaluation/io.py`, `evaluation/report_tables.py`, `evaluation/report_numbers.py`, `models/baselines.py`, `control/*` (CTM, ALINEA, predictive ALINEA, experiment), `scripts/run_baselines.py`, `scripts/evaluate_all.py`, `scripts/evaluate_ablations.py`, `scripts/hybrid_analysis.py`, `scripts/run_control.py`, `scripts/make_report_assets.py`.
- **Tests:** `test_metrics_masking.py`, `test_baselines.py`, `test_ctm.py`.
- **Experiments:** baselines, all metric tables, Diebold-Mariano and bootstrap results, hybrid gain decomposition, ablation evaluation, the simulated ramp-metering study (disclaimer in every output), ADR-005.
- **Report sections:** Phase 1 VII (Experimental Setup) and results tables; Phase 2 VI (Experimental Setup), VII (tables, significance), X (Simulated Control).
- **Theory notes:** `metrics.md`, `significance_testing.md`, `control.md`.

## 4. Kanchan
- **Modules:** `data/load.py`, `data/graph.py`, `data/missing.py`, `data/split.py`, `data/scaling.py`, `data/windows.py`, `data/pipeline.py`, `features/temporal.py`, `viz/eda_plots.py`, `scripts/download_data.py`, `scripts/prepare_data.py`, `scripts/run_eda.py`, `scripts/make_eda_summary.py`, `scripts/run_ablations.py`, `scripts/run_ablation_chain.sh`.
- **Tests:** `test_load.py`, `test_graph.py`, `test_split_no_leakage.py`, `test_scaling_train_only.py`, `test_windows.py`.
- **Experiments:** data download and integrity checks, graph rebuild and statistics, missing-data analysis, the ten EDA figures and the propagation evidence, ablations A1-A12 (runs and write-up), ADR-002, 003, 004.
- **Report sections:** Phase 1 III (Dataset and EDA), IV (Problem Formulation); Phase 2 III (Data, Graph and Regularisation), VIII (Ablations).
- **Theory notes:** `chronological_splits.md`, `missing_data.md`.

## 5. Vaibhav (lightest, most mechanical)
- **Modules:** repository scaffold (`pyproject.toml`, `Makefile`, `.github/workflows/ci.yml`, `.gitignore`, `configs/base.yaml` upkeep), `viz/congestion_map.py`, `viz/horizon_plots.py`, `viz/attention_viz.py`, `scripts/make_maps.py`, `scripts/make_docs.py`, `scripts/run_phase2_chain.sh`, `README.md`, `slides/phase1.md`, `slides/phase2.md`, `report/refs.bib` (verifying every entry), LaTeX assembly of both reports.
- **Tests:** keeps `make lint` and `make test` green (CI), `test_config.py`, `test_end_to_end_debug.py`.
- **Experiments:** runs the full pipeline with the `make` targets, produces the interactive maps, error maps, space-time diagram and attention figures, keeps `docs/experiment_log.md` current.
- **Report sections:** Phase 1 I, II, IX, X; Phase 2 I, II, IX (Interpretability and Maps), XI (Limitations and Ethics); references; final PDF builds.
- **Theory notes:** `random_forest.md`, `diffusion_conv.md`.
- **Docs:** `docs/definition_of_done.md`, `docs/viva/README.md`.

## Difficulty-order rationale
1. The GAT layer, STGAT and trainer carry the highest technical risk (numerical equivalence with PyG, GPU training under compute limits).
2. Tree models and the leakage-safe out-of-fold hybrid are broad and full of leakage pitfalls.
3. The evaluation and statistics code (HAC Diebold-Mariano, block bootstrap) and the control simulator are small but easy to get subtly wrong.
4. The data pipeline, EDA and ablation runs follow clear recipes.
5. Visualisation, documentation and report assembly are the most mechanical.

## Git hygiene
Commits should come from each member's own account on their owned modules (pull requests reviewed by Pranav). The initial history was bootstrapped from one account; going forward use feature branches named `<member>/<topic>`.
