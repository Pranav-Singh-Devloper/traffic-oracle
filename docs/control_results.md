# Simulated control results (auto-generated)

> This is an offline simulation on a synthetic network. It demonstrates how forecasts could inform a control policy. No live or real-world traffic control is performed or claimed.

Assumptions: [control_assumptions.md](control_assumptions.md). Mean over 5 test weekdays; standard deviations are in `results/metrics/control_summary.csv`.

| policy | TTS_veh_h | distance_veh_mi | mainline_avg_speed_mph | max_ramp_queue_veh | congestion_cell_minutes |
|---|---|---|---|---|---|
| no_control | 2704.8 | 139350.3 | 52.4 | 0.0 | 399.2 |
| fixed_time | 2704.8 | 139350.3 | 52.4 | 0.0 | 399.2 |
| alinea | 2705.5 | 139644.5 | 57.1 | 221.3 | 243.3 |
| predictive_alinea | 4783.5 | 137226.1 | 62.0 | 766.0 | 78.6 |
| oracle_predictive_alinea | 3854.6 | 137879.5 | 61.3 | 607.0 | 97.9 |
| lookahead_alinea | 2714.7 | 139656.5 | 58.2 | 243.9 | 201.8 |
| oracle_lookahead_alinea | 2728.3 | 139647.8 | 58.1 | 248.4 | 204.2 |

Reading guide: lower TTS, queue and congested cell-minutes are better; higher mainline speed is better.

Figures: `results/figures/control/`.
