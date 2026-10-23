# Significance testing

## Diebold–Mariano (DM)
Compare two forecasts' loss series `d_t = L(e1_t) − L(e2_t)` (absolute error averaged over sensors, per horizon). Under H0, `E[d_t] = 0`. Statistic `DM = d̄ / sqrt(V̂/T)` where V̂ is a long-run variance.

## Why HAC
Test anchors overlap (stride 1) and errors are persistent, so `d_t` is autocorrelated. A naive variance is too small, giving false significance. We use a Newey–West (Bartlett) variance with L = max(h−1, ⌈T^{1/3}⌉) lags, the Harvey–Leybourne–Newbold small-sample correction and a t(T−1) reference. The classic rule uses h−1 lags (forecast errors at horizon h are MA(h−1)); we take the larger lag for robustness.

## Block bootstrap
Resample whole days (blocks of 288 steps) with replacement, recompute MAE; the 2.5/97.5 percentiles give a 95% CI. For paired differences the same days are resampled for both models, so day-level difficulty cancels.

## Caveats
Multiple comparisons (many horizons/slices) inflate false positives; we report effect sizes with CIs. DL models use 3 seeds: DM is run per seed pair and we report the median statistic and the largest p-value.

## Likely viva questions
1. *Why not a paired t-test?* Autocorrelated losses.
2. *What does a negative DM statistic mean here?* Model 1 has the lower loss.
3. *Why blocks of one day?* Daily seasonality is the dominant dependence.
