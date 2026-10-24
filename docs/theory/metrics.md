# Metrics

**Owner:** Himanshu

All metrics are computed in **mph**, on **observed** targets only (mask = speed > 0).

- **MAE** `= Σ m|y−ŷ| / Σ m`: average error in mph; robust, same units as speed.
- **RMSE** `= sqrt(Σ m(y−ŷ)² / Σ m)`: penalises large errors more; larger than MAE whenever errors vary. A big RMSE/MAE gap signals occasional big misses (e.g. missing the onset of congestion).
- **MAPE** `= 100·Σ m'|y−ŷ|/y / Σ m'`, `m' = m ∧ (y ≥ 1 mph)`: relative error; unstable at low speeds because the denominator is small, and asymmetric (over-prediction costs more than under-prediction). Use it as secondary.

## Why per horizon
Error grows with horizon; a single averaged number hides where models differ. Headline horizons are 15/30/60 min (steps 3/6/12); all 12 are in `all_metrics_long.csv`.

## Why slices
Congestion is rare overall but frequent at peaks; averaged metrics are dominated by easy free-flow samples. We also report peak, off-peak weekday, weekend, congested (<40 mph) and free-flow.

## Edge cases
An all-masked slice returns NaN, never 0. NaN predictions (untrained horizons) are removed from the mask.

## Likely viva questions
1. *Why masked metrics?* Zero readings are outages, counting them would reward/penalise models for nothing.
2. *MAE vs RMSE choice?* Report both; MAE is our training/early-stopping metric.
3. *Why not MAPE as headline?* Instability at low speed.
4. *Why is historical average flat across horizons?* It ignores the current state, so its error does not depend on horizon.
