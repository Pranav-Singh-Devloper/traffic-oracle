# XGBoost

**Owner:** Kunal

## Model
Additive trees: `ŷ_i = Σ_{k=1}^K f_k(x_i)`, each `f_k` a regression tree with leaf weights `w`.

## Objective and second-order approximation
`L = Σ_i l(y_i, ŷ_i) + Σ_k Ω(f_k)`, `Ω(f) = γT + ½λ‖w‖²` (T leaves). At step t, expand the loss to second order around the current prediction:
`L^(t) ≈ Σ_i [g_i f_t(x_i) + ½ h_i f_t(x_i)²] + Ω(f_t)`, with `g_i = ∂l/∂ŷ`, `h_i = ∂²l/∂ŷ²`.
For a leaf containing index set I the optimal weight is `w* = −G/(H+λ)` (G = Σg, H = Σh) with score `−½ G²/(H+λ)`.

## Split gain
`Gain = ½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ`
A split is made only if gain > 0: γ is a complexity price per leaf, λ shrinks leaf weights.

## Practical details used here
- `tree_method="hist"` (histogram binning, fast), early stopping on validation MAE.
- Objective is chosen by Optuna from squared error / absolute error / pseudo-Huber; MAE is the reported metric, and absolute error targets the conditional median.
- **Per-segment vs global:** per-segment trains one model per sensor (207 x horizons, each ~24k rows, no sharing). Global pools sensors (~5M rows) with sensor id and static features, borrowing strength across sensors but still without neighbor information.

## Likely viva questions
1. *Why second order?* Newton step gives faster, more stable fitting and a closed-form leaf weight.
2. *What do λ and γ do?* λ shrinks leaf weights; γ prunes splits with small gain.
3. *Difference between gain importance and SHAP?* Gain is split-based and biased toward high-cardinality/early splits; SHAP attributes each prediction consistently (we use TreeSHAP via `pred_contribs`).
4. *Why can XGBoost not model propagation?* A row only contains that sensor's own history; neighbor information must be hand-engineered (XGB-spatial).
5. *Why do trees handle `speed==0`-like issues badly?* We impute inputs and add missing flags; targets are masked.
