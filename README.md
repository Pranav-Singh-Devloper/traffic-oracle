# Traffic Flow Prediction & Simulated Signal Control

Graph Attention Networks vs. segment-level ML on METR-LA. See [SPEC.md](SPEC.md) and the build plan in [phases/](phases/00_OVERVIEW.md).

Control experiments are offline simulation only; no real-world traffic control is performed or claimed.

## Run
```
make data && make all
```
Runtimes / hardware: _filled in from docs/experiment_log.md_.

## Results
<!-- RESULTS_START -->
### Test error by horizon (mph, masked)

| model | MAE@15min | MAE@30min | MAE@60min | RMSE@15min | RMSE@30min | RMSE@60min | MAPE@15min | MAPE@30min | MAPE@60min |
|---|---|---|---|---|---|---|---|---|---|
| historical average | 4.19 | 4.19 | 4.19 | 7.86 | 7.86 | 7.86 | 13.06 | 13.06 | 13.06 |
| persistence | 3.49 | 4.20 | 5.37 | 6.39 | 8.02 | 10.30 | 8.60 | 10.83 | 14.48 |
| rf multioutput | 3.14 | 3.80 | 4.73 | 5.70 | 6.89 | 8.25 | 8.50 | 10.87 | 14.27 |
| seasonal naive | 5.50 | 5.50 | 5.50 | 10.71 | 10.71 | 10.71 | 15.57 | 15.57 | 15.57 |
| xgb global | 2.88 | 3.33 | 3.78 | 5.69 | 6.80 | 7.76 | 7.87 | 9.90 | 12.11 |
| xgb per segment | 2.86 | 3.28 | 3.68 | 5.60 | 6.62 | 7.52 | 7.92 | 9.77 | 11.65 |
| xgb spatial | 2.82 | 3.26 | 3.73 | 5.46 | 6.54 | 7.54 | 7.63 | 9.53 | 11.82 |

### MAE at 60 min by slice (mph)

| model | offpeak_weekday | peak | peak_am | peak_pm | weekend |
|---|---|---|---|---|---|
| historical_average | 3.51 | 6.06 | 4.93 | 7.13 | 3.50 |
| persistence | 4.85 | 7.45 | 6.93 | 7.95 | 3.98 |
| rf_multioutput | 3.82 | 7.24 | 6.52 | 7.91 | 3.80 |
| seasonal_naive | 4.37 | 7.48 | 6.18 | 8.70 | 5.83 |
| xgb_global | 3.11 | 5.65 | 4.67 | 6.58 | 3.05 |
| xgb_per_segment | 3.11 | 5.26 | 4.19 | 6.28 | 3.09 |
| xgb_spatial | 3.08 | 5.54 | 4.65 | 6.38 | 3.03 |
<!-- RESULTS_END -->
