# Phase 7 — Hybrid Models (M7)

**Goal:** a leakage-safe residual hybrid (XGBoost base + STGAT residual), a gating ensemble, and a simple average — each compared against both components with significance tests.
**SPEC refs:** §10, §8.4, §14 M7.
**Prerequisites:** Phases 4 (XGB-global) and 6 (STGAT, trainer).
**Owner:** Kunal (models, OOF); Himanshu (significance, gain analysis)

---

## 7.1 Why the hybrid exists (put this in the docstring of `hybrid.py`)

XGBoost is good at local autoregression and calendar effects but sees no neighbors. STGAT is trained to learn **only what XGBoost misses**: `r = y − ŷ_XGB`. If propagation drives that residual, the graph model should reduce it most at peaks and long horizons.

`ŷ_hybrid(t+h, i) = ŷ_XGB(t+h, i) + g_θ(X_window, ŷ_XGB, G)(h, i)`

## 7.2 Base model choice

Use `xgb_global` trained on **all 12 horizons** (12 models) so the residual exists for every horizon. Record in an ADR why `xgb_global` (not per-segment): one model per horizon makes OOF training 5× cheaper and its accuracy is similar (cite the Phase 4 table). If the team prefers per-segment, the code must accept either.

## 7.3 Out-of-fold predictions — `src/traffic/models/hybrid.py::make_oof_predictions`

**Never use in-sample XGBoost predictions on train** — they are overfit, so residuals look unrealistically small and STGAT learns to correct almost nothing.

Forward-chaining CV on the **train** split:
1. Split train anchors into 5 contiguous chronological blocks `B1…B5`.
2. For k = 2..5: fit XGB (same tuned hyperparameters, early stopping on the *last 10% of the training part*, never on block k) on `B1 … B(k−1)`; predict block k.
3. Block 1 has no OOF prediction → **drop B1** from residual training (default). Alternative: a warm-up model trained on the first half of B1 predicting the second half. Record choice as ADR.
4. **Gap between train blocks and predicted block:** leave a gap of `H + F` steps at each block boundary so no training window's target overlaps the predicted block's inputs.
5. Final XGB = trained on the full train split (Phase 4 model) → predicts val and test.

Output: `results/predictions/xgb_oof_train.npz` (`[S_train, F, N]`, NaN where unavailable) + val/test predictions.

Known caveat to state in the report: OOF models for early blocks see less data than the final model, so OOF residuals are slightly *larger* than the val/test residuals. That is the conservative direction.

## 7.4 Residual model

- Inputs: the usual `X [B, H, N, C]` **plus** `ŷ_XGB [B, F, N]` (scaled) fed to the decoder: concat `ŷ_XGB` (transposed to `[B, N, F]`) to the readout features before the MLP.
- Target: `r = scale(y) − scale(ŷ_XGB)`, masked where y missing **or** ŷ_XGB is NaN.
- Loss: masked MAE on the residual. Early stop on **val** hybrid MAE (in mph).
- Initialize the decoder's last layer to zero so the hybrid starts as exactly XGB — makes training stable and easy to explain.
- 3 seeds.
- `scripts/train_hybrid.py --kind residual|gating|average [--debug]`.

## 7.5 Gating ensemble (ablation, SPEC §10.2)

`ŷ = w·ŷ_STGAT + (1 − w)·ŷ_XGB`, with `w = σ(MLP(horizon one-hot, peak_flag, tod_sin, tod_cos, window_missing_rate))`.
- Train the gate on the **first 80% of val** predictions; select/early-stop on the **last 20% of val**. STGAT and XGB model selection already used val, so state this reuse as a limitation in the ADR.
- Report the learned `w` by horizon and peak/off-peak (figure). Does it trust STGAT more at long horizons and peaks?

## 7.6 Simple average

`ŷ = 0.5·ŷ_XGB + 0.5·ŷ_STGAT` (STGAT seed-averaged or per seed — be consistent and say which).

## 7.7 Evaluation

- All hybrid variants into `main_table.csv` (mean ± std over seeds).
- DM tests: hybrid vs `xgb_global`, hybrid vs `stgat`, hybrid vs average — per headline horizon → `significance.csv`.
- Gain decomposition: `ΔMAE` vs each component by horizon × peak/off-peak × congested/free-flow → `results/metrics/hybrid_gain.csv` + figure.
- `docs/hybrid_analysis.md`: where does the gain come from? If the hybrid does **not** beat STGAT, say so and give the likely reason (e.g. STGAT already captures the local structure).

## 7.8 Tests

- `test_oof_no_leakage.py` (add): for each fold, `max(train target_ts) + gap < min(predicted input_ts)`; block 1 has no OOF predictions.
- Zero-initialized residual head ⇒ hybrid prediction == XGB prediction before training (atol 1e-5).

---

## Acceptance (M7)

- [ ] OOF predictions generated with forward-chaining CV; leakage test passes
- [ ] Residual hybrid, gating, and average rows in `main_table.csv`
- [ ] DM tests vs both components in `significance.csv`
- [ ] Gain-decomposition table + `docs/hybrid_analysis.md`, honest either way

**Commit:** `M7: OOF XGB, residual and gating hybrids, significance`
**STOP for review.**
