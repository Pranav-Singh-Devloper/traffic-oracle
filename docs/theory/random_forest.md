# Random Forest

## Idea
Average many decorrelated, deep trees. Each tree is fit on a bootstrap sample (`max_samples`) and each split considers a random subset of features (`max_features`).

## Variance reduction
For B trees with variance σ² and pairwise correlation ρ: `Var(mean) = ρσ² + (1−ρ)σ²/B`. Bagging shrinks the second term; feature subsampling lowers ρ, which is the floor of the variance. Bias stays about that of one deep tree.

## Native multi-output
`RandomForestRegressor` fits one tree per bootstrap sample for all outputs at once: a split minimises the **sum of squared errors over all 12 horizons**, so a leaf stores a 12-vector. Predictions across horizons are coherent and one model replaces 12, but the split criterion is dominated by long horizons (larger errors).

## Choices here
- Pooled over sensors, rows with any missing horizon target dropped (share logged), training rows subsampled (see ADR-006) for memory/time.
- Squared-error criterion (MAE is evaluated, so RF is not tuned for the metric directly).

## Likely viva questions
1. *Why doesn't adding trees overfit?* Variance falls monotonically with B (bias is unchanged).
2. *Why max_features < all?* Decorrelates trees (lowers ρ).
3. *Why does RF extrapolate poorly?* Leaves output training-set averages, so predictions stay in the observed range.
4. *Multi-output pros/cons?* Coherent horizons and speed vs. no horizon-specific tuning.
