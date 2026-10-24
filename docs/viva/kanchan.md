# Viva prep: Kanchan (rank 4): data pipeline, graph, missing data, EDA, ablations

Team T042. Also answer `shared_questions.md` (10 questions everyone must know). Ownership: see `CONTRIBUTIONS.md`.

1. **What are the dataset facts you assert on load?** 207 sensors, 34,272 5-min steps (119 days), 2012-03-01 to 2012-06-27, regular index, no NaN, non-negative (`data/load.py`).
2. **Why did `pd.read_hdf` fail and what did you do?** Old pandas fixed-format attributes are bytes; with new pandas/PyTables it raises TypeError. We fall back to PyTables (ADR-004); data identical, checks still pass.
3. **Why normalise sensor IDs to strings?** pkl, txt, csv and h5 mix int/str ids; mismatches silently misalign columns.
4. **How do you guarantee column order is consistent?** All arrays use `adj_mx.pkl` order; loader asserts equality or reorders and logs.
5. **How is sigma computed in the graph rebuild?** Std of finite entries of the N×N distance matrix (not of all CSV rows), exactly as DCRNN.
6. **What does `W` vs `Wᵀ` mean physically?** Row i aggregates from column j. `W` built from distance i→j ⇒ downstream senders; `Wᵀ` upstream.
7. **Why keep the pkl as canonical if you rebuilt it?** Rebuild proves understanding; pkl is the published artifact (they agree to 1e-7).
8. **What graph statistics did you find?** See `results/metrics/graph_stats.json`: edges, components, diameter, degree histograms.
9. **What are the imputation options and the default?** (a) forward-fill ≤6 steps then train time-of-day mean + flag (default); (b) leave zeros + flag (ablation A9).
10. **Why can forward-fill cross the split boundary?** It only uses the past, so no leakage.
11. **How is the scaler fit and why exclude zeros?** Masked mean/std over training observed values only; zeros would bias both.
12. **What happens to missing inputs after scaling under zero_mask?** Set to 0 in scaled space (train mean) and carry an is_missing channel (ADR-003).
13. **How do you build windows without a Python loop and without crossing splits?** Anchors t∈[start+H−1, stop−F−1] per split; arrays sliced by index (`data/windows.py`).
14. **What does the leakage test assert?** max(train target ts) < min(val input ts) and max(val target) < min(test input).
15. **What does the scaling test assert?** Changing val/test values does not change scaler parameters or time-of-day means.
16. **Why is cyclic encoding used for time?** So 23:55 and 00:00 are close; one-hot/linear hours break continuity.
17. **What did the lagged cross-correlation show?** Neighbors correlate strongly; direction evidence is modest (more positive than negative lags at peaks); see `docs/eda_summary.md`.
18. **How were the example pairs in the xcorr figure chosen?** By stated rules (top correlation; largest non-zero lag among corr ≥ 0.5), plus a histogram over all pairs to avoid cherry-picking.
19. **What distribution shift exists between splits?** Test has a higher outage rate than train/val.
20. **How would you add PeMS flow/occupancy?** Feature builders take extra channels; lags/rolling stats generalise per channel.
21. **What is over-smoothing and how do you test it?** Representations of neighbors converge with depth; ablation A6 (1/2/3 layers).
22. **How is the ablation budget reduced and why is it still fair?** Short epochs, one seed, with a same-budget default run as the reference.
23. **What does ablation A2 (random graph) test?** Whether road topology matters or just extra parameters.
24. **How are H=6/24 compared fairly?** On the intersection of test target timestamps.
25. **Why bidirectional adjacency as default?** More information and DCRNN also uses both directions; A3 tests directions separately.
