# Viva prep: Member D — evaluation, significance, hybrid, maps, control

_Name: TODO (fill in from CONTRIBUTIONS.md). Also answer `shared_questions.md`._

1. **What is the common prediction format?** npz with y_pred/y_true/mask [S,F,N], target_ts, sensor ids, meta; alignment asserted across models.
2. **How are slices defined?** Peak = weekday AM/PM windows on target timestamp; off-peak weekday; weekend; congested (<40 mph) vs free-flow.
3. **Why is MAPE thresholded at 1 mph?** Small denominators blow it up (ADR-005).
4. **What does an all-masked slice return?** NaN with a warning, never 0.
5. **Explain the DM test implementation.** Loss differential of sensor-averaged absolute errors; Bartlett long-run variance, HLN correction, t(T−1).
6. **What does the sign of the DM statistic mean here?** Negative: model A (first) has lower loss.
7. **How does the block bootstrap work?** Resample whole days with replacement and recompute MAE.
8. **What sanity checks run automatically?** MAE@60 < 3.0 ⇒ leakage warning; not beating persistence at 15 min ⇒ broken; misalignment ⇒ error.
9. **How is the OOF procedure done?** 5 chronological train blocks; block k predicted by a model trained on earlier blocks with a gap; block 1 dropped.
10. **Why zero-initialise the residual head?** Hybrid starts as XGBoost; training can only add improvements.
11. **How is the gate trained and what is the caveat?** First 80% of val train, last 20% early-stop; val no longer pure for model selection.
12. **What would it mean if the hybrid doesn't beat STGAT?** STGAT already captures local structure; report honestly.
13. **How do you pick the AM-peak map timestamp?** Programmatically: test weekday 07–09 target with most congested sensors; saved to JSON.
14. **What does the space-time diagram show?** Speed along a corridor over time; truth vs forecasts; backward-moving fronts if present.
15. **How is the corridor chosen?** Longest chain following nearest downstream neighbors with increasing road distance.
16. **How is attention summarised?** Shares on self / downstream / upstream / both-direction links, AM peak vs night; qualitative only.
17. **State the CTM update.** Sending/receiving functions with triangular FD; flow = min(send, receive); merge priority.
18. **What is the CFL condition?** v_free·dt ≤ cell length.
19. **What is ALINEA and its predictive variant?** Integral occupancy feedback; predictive uses max(measured, forecast-implied occupancy).
20. **What are the honesty limits of the control experiment?** Synthetic demand shaped by speed, open loop, simulation only (disclaimer everywhere).
