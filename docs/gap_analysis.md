# Gap analysis: do neighbour features help? (auto-generated)

`delta_mae = MAE(xgb_global) - MAE(xgb_spatial)`; positive means the hand-made spatial features help.
95% intervals are paired day-block bootstrap; `p_value` is Diebold-Mariano (HAC).

| horizon | slice | mae_global | mae_spatial | delta_mae | ci_lo | ci_hi | p_value |
|---|---|---|---|---|---|---|---|
| 3 | all | 2.880 | 2.820 | 0.059 | 0.051 | 0.069 | 0.000 |
| 3 | peak | 3.866 | 3.749 | 0.117 | 0.101 | 0.135 | 0.000 |
| 3 | offpeak_weekday | 2.579 | 2.540 | 0.038 | 0.031 | 0.048 | 0.000 |
| 3 | weekend | 2.356 | 2.320 | 0.036 | 0.025 | 0.057 | 0.000 |
| 3 | congested | 7.513 | 7.177 | 0.336 | 0.282 | 0.392 | 0.006 |
| 3 | freeflow | 2.226 | 2.205 | 0.020 | 0.017 | 0.024 | 0.000 |
| 6 | all | 3.331 | 3.258 | 0.073 | 0.059 | 0.090 | 0.000 |
| 6 | peak | 4.730 | 4.578 | 0.152 | 0.126 | 0.182 | 0.000 |
| 6 | offpeak_weekday | 2.863 | 2.816 | 0.048 | 0.034 | 0.065 | 0.000 |
| 6 | weekend | 2.697 | 2.661 | 0.037 | 0.024 | 0.057 | 0.000 |
| 6 | congested | 10.147 | 9.588 | 0.559 | 0.452 | 0.670 | 0.000 |
| 6 | freeflow | 2.369 | 2.364 | 0.005 | -0.003 | 0.012 | 0.764 |
| 12 | all | 3.781 | 3.730 | 0.051 | 0.033 | 0.072 | 0.000 |
| 12 | peak | 5.654 | 5.542 | 0.112 | 0.081 | 0.143 | 0.000 |
| 12 | offpeak_weekday | 3.113 | 3.081 | 0.032 | 0.012 | 0.055 | 0.000 |
| 12 | weekend | 3.050 | 3.030 | 0.020 | 0.000 | 0.041 | 0.062 |
| 12 | congested | 13.034 | 12.576 | 0.458 | 0.343 | 0.570 | 0.001 |
| 12 | freeflow | 2.475 | 2.482 | -0.007 | -0.018 | 0.004 | 0.038 |

**Reading.** Spatial features help in almost every slice, most at peaks and in congested states, and the effect does not grow with the horizon (it is largest around 30 minutes). That supports "neighbours help during congestion" but only partially supports "graph benefit grows with horizon".

## Daily-lag side experiment (not in the main table)

`lag_288` (same time yesterday) added to xgb_global:

| model | horizon | MAE | MAPE | RMSE | MAE_std | MAPE_std | RMSE_std | horizon_min |
|---|---|---|---|---|---|---|---|---|
| xgb_global_daily | 3 | 2.879 | 7.882 | 5.685 | - | - | - | 15 |
| xgb_global_daily | 6 | 3.330 | 9.881 | 6.795 | - | - | - | 30 |
| xgb_global_daily | 12 | 3.785 | 12.099 | 7.758 | - | - | - | 60 |

Compare with `xgb_global` in `results/metrics/main_table.csv`; the daily lag adds essentially nothing once the 12-step history, time-of-day features and sensor identity are present.
