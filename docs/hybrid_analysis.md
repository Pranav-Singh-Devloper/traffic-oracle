# Hybrid analysis (auto-generated)

## Gain of the residual hybrid over its components (mph, positive = hybrid better)
| component | slice | 15 | 30 | 60 |
|---|---|---|---|---|
| stgat | all | 0.081 | 0.058 | 0.019 |
| stgat | congested | 0.258 | 0.327 | -0.591 |
| stgat | freeflow | 0.056 | 0.020 | 0.105 |
| stgat | offpeak_weekday | 0.073 | 0.047 | 0.056 |
| stgat | peak | 0.106 | 0.084 | -0.091 |
| stgat | weekend | 0.069 | 0.049 | 0.064 |
| xgb_global | all | 0.059 | 0.113 | 0.133 |
| xgb_global | congested | 0.609 | 1.283 | 1.536 |
| xgb_global | freeflow | -0.018 | -0.052 | -0.065 |
| xgb_global | offpeak_weekday | 0.027 | 0.041 | 0.037 |
| xgb_global | peak | 0.146 | 0.304 | 0.374 |
| xgb_global | weekend | 0.027 | 0.046 | 0.066 |

## Learned gate: mean weight on STGAT (test set)
| horizon | mean_w_stgat_peak | mean_w_stgat_offpeak |
|---|---|---|
| 1.000 | 0.349 | 0.178 |
| 2.000 | 0.493 | 0.274 |
| 3.000 | 0.558 | 0.328 |
| 4.000 | 0.653 | 0.356 |
| 5.000 | 0.724 | 0.397 |
| 6.000 | 0.786 | 0.450 |
| 7.000 | 0.823 | 0.488 |
| 8.000 | 0.846 | 0.503 |
| 9.000 | 0.848 | 0.518 |
| 10.000 | 0.844 | 0.488 |
| 11.000 | 0.825 | 0.455 |
| 12.000 | 0.789 | 0.416 |

**Reading.** Over XGBoost-global the residual hybrid's gain grows with horizon and is largest in congested states and at peaks, where per-sensor history is least informative; in free flow it is slightly negative at 60 minutes. Over STGAT the gain is largest at 15 minutes and shrinks with horizon: STGAT already captures much of the local structure at long range. The gate puts more weight on STGAT at peaks and at longer horizons, as hypothesised. Significance (DM) is in `results/metrics/significance.csv`. Caveat: the gate used the last part of validation for early stopping (ADR-008).
