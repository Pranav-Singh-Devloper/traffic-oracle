# Viva prep: Member B — features, Phase 1 ML, SHAP, gap analysis

_Name: TODO (fill in from CONTRIBUTIONS.md). Also answer `shared_questions.md`._

1. **List the tabular features.** Lags 0–11, rolling mean/std/min/max (3,6,12), diffs, missing flags, cyclic time, weekend/holiday, static sensor features (pooled).
2. **Why is the daily lag off by default?** It uses data outside the 12-step window, breaking the equal-history comparison; run as a separate experiment.
3. **Why drop rows with missing targets?** Equivalent to masking; training on zeros would teach outages.
4. **Why XGBoost per segment AND pooled?** Per-segment: sensor-specific but data-poor; pooled: shares strength but needs identity features. Neither sees neighbors.
5. **Why was absolute error selected as the objective?** Optuna picked it on validation MAE; it targets the median and matches the metric.
6. **How was tuning kept leak-free?** Validation only; test never touched; budget logged in tuning CSVs.
7. **What was reduced for laptop limits?** Trials (15/6), pooled models on 25% rows, RF on 10% rows, trees capped at 400; documented in the report.
8. **Explain XGBoost's gain formula.** ½[G_L²/(H_L+λ)+G_R²/(H_R+λ)−(G)²/(H+λ)]−γ.
9. **What do λ and γ do?** Leaf-weight shrinkage and split pruning.
10. **Why is RF multi-output and what is the caveat?** One forest predicts 12 horizons with a joint squared-error split criterion; rows with any missing horizon are dropped.
11. **How are neighbor features defined?** Row-normalised W (downstream) and Wᵀ (upstream) applied to lagged speeds; 2-hop mean; own minus neighbor.
12. **What happens for sensors without a neighbor in a direction?** NaN; XGBoost handles it natively.
13. **How is SHAP computed?** XGBoost's `pred_contribs` (TreeSHAP) on 5,000 seeded test rows.
14. **What does SHAP show about horizons?** Short-window means/latest speed dominate at 15 min; sensor identity and time-of-day matter more at 60 min.
15. **Summarise the gap analysis.** Spatial features help consistently, more at peaks/congestion; gain does not grow with horizon (`results/metrics/gap_analysis.csv`).
16. **Why paired block bootstrap for the gap?** Resampling the same days for both models cancels day difficulty.
17. **Why might per-segment beat spatial at 60 min?** Pooled models were trained on subsampled rows; per-segment uses all of each sensor's data.
18. **Which sensors are hardest?** A cluster near the same location (see `hardest_sensors_xgb_global_h12.csv`).
19. **How do you check for leakage in features?** `test_features_no_future.py`: perturbing the future leaves features unchanged.
20. **How are predictions mapped back to [S,F,N]?** Row order anchor-major; tested in `test_tabular_alignment.py`.
