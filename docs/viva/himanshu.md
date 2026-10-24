# Viva prep: Himanshu (rank 3): evaluation, statistics, simulated control

Also answer `shared_questions.md` (10 questions everyone must know). Ownership: see `CONTRIBUTIONS.md`.

1. **What is the common prediction format?** npz with y_pred/y_true/mask [S,F,N], target_ts, sensor ids, meta; alignment asserted across models.
2. **How are slices defined?** Peak = weekday AM/PM windows on target timestamp; off-peak weekday; weekend; congested (<40 mph) vs free-flow.
3. **Why is MAPE thresholded at 1 mph?** Small denominators blow it up (ADR-005).
4. **What does an all-masked slice return?** NaN with a warning, never 0.
5. **Explain the DM test implementation.** Loss differential of sensor-averaged absolute errors; Bartlett long-run variance, HLN correction, t(T−1).
6. **What does the sign of the DM statistic mean here?** Negative: model A (first) has lower loss.
7. **How does the block bootstrap work?** Resample whole days with replacement and recompute MAE.
8. **What sanity checks run automatically?** MAE@60 < 3.0 ⇒ leakage warning; not beating persistence at 15 min ⇒ broken; misalignment ⇒ error.
9. **State the CTM update.** Sending/receiving functions with triangular FD; flow = min(send, receive); merge priority.
10. **What is the CFL condition?** v_free·dt ≤ cell length.
11. **What is ALINEA and its predictive variant?** Integral occupancy feedback; predictive uses max(measured, forecast-implied occupancy).
12. **What are the honesty limits of the control experiment?** Synthetic demand shaped by speed, open loop, simulation only (disclaimer everywhere).
13. **How does the baseline suite protect against weak comparisons?** Persistence, historical average and seasonal naive are evaluated with the same masks and timestamps; every learned model must beat persistence at 15 minutes (automatic warning otherwise).
14. **Why report results per horizon and per slice?** Error grows with horizon and congestion is rare overall but common at peaks; averages hide where models differ (`docs/theory/metrics.md`).
15. **What do the bootstrap confidence intervals tell you that p-values do not?** They give the size and uncertainty of a difference in mph, resampling whole days so day-level difficulty cancels in paired comparisons.
16. **Why is the Diebold-Mariano test applied per seed pair?** A single seed's errors depend on its random init; pairing seeds avoids comparing a seed's luck against a different seed's luck; we report the median statistic and the largest p-value.
17. **What does the gain-decomposition table show?** MAE(component) minus MAE(hybrid) by horizon and slice; the residual hybrid helps most in congested states at long horizons.
