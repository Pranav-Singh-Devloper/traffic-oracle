# Hybrid: residual learning with out-of-fold predictions

## Rationale
XGBoost captures local autoregression and calendar effects cheaply. Spatial propagation is its blind spot (Phase 1 gap analysis). We train a graph model on the **residual** `r = y − ŷ_XGB`, giving it ŷ_XGB as an input, so it concentrates on what XGBoost misses. The last decoder layer is zero-initialised, so training starts exactly at the XGBoost forecast.

## Why out-of-fold
XGBoost's predictions on its own training rows are overfit: residuals there are unrealistically small, so the graph model would learn to correct almost nothing and then face larger residuals on val/test (train/test mismatch). Out-of-fold (OOF) predictions have realistic error magnitudes.

## Forward-chaining OOF (time series)
Split train into 5 chronological blocks. For block k ≥ 2, train on blocks < k (leave a gap of H+F−1 anchors) and predict block k. Block 1 has no OOF forecast and is dropped from residual training. Random K-fold would leak the future into the models.
Caveat: early-block models are trained on less data, so their OOF errors are a bit larger than the final model's val/test errors (conservative direction).

## Stacking view
This is stacking with a time-aware base-learner protocol: level-0 = XGBoost, level-1 = graph residual learner trained on OOF level-0 outputs.

## Gating ensemble
`ŷ = w·ŷ_STGAT + (1−w)·ŷ_XGB`, `w = σ(MLP(horizon one-hot, peak flag, tod, missing rate))`, trained on the first 80% of val and early-stopped on the last 20% (val is then no longer pure for model selection; documented limitation).

## Likely viva questions
1. *Why not use in-sample predictions?* See above.
2. *Why forward-chaining?* Respect time order; no future in training.
3. *What if the hybrid does not beat STGAT?* Report it: STGAT may already capture the local structure.
4. *Why zero-init?* Start at a strong baseline; training only adds improvements.
