# Viva prep: Vaibhav (rank 5): visualisation, documentation, report assembly

Also answer `shared_questions.md` (10 questions everyone must know). Ownership: see `CONTRIBUTIONS.md`.

1. **How do you pick the AM-peak map timestamp?** Programmatically: test weekday 07–09 target with most congested sensors; saved to JSON.
2. **What does the space-time diagram show?** Speed along a corridor over time; truth vs forecasts; backward-moving fronts if present.
3. **How is the corridor chosen?** Longest chain following nearest downstream neighbors with increasing road distance.
4. **How is attention summarised?** Shares on self / downstream / upstream / both-direction links, AM peak vs night; qualitative only.
5. **How is the full pipeline reproduced?** `make data`, `make eda`, `make phase1`, then the chain scripts for ML, DL, hybrids/maps/control and ablations; `make report` builds both PDFs.
6. **How do report numbers stay consistent with results?** `scripts/make_report_assets.py` writes `report/tables/numbers.tex` macros from the CSVs; the prose uses macros, not typed numbers.
7. **What does CI run and what is skipped without data?** Lint plus tests; tests marked `real_data` are skipped when `data/raw/metr-la.h5` is absent.
8. **Which files are git-ignored and why?** Dataset, predictions, checkpoints and logs are large and regenerable; only code, configs, CSV metrics and figures are tracked.
9. **Why was the report built with tectonic?** No LaTeX distribution was installed; tectonic is a small self-contained engine that fetches packages on demand.
10. **What is the colour scheme of the congestion maps?** Red below 40 mph, orange 40-55, green above 55, grey for missing.
11. **How do you check the BibTeX entries?** Each entry's authors, venue and year are verified against the paper; DOIs are only added when confirmed (never invented); the spec's CityFlow attribution was corrected.
12. **What is the Marp slide workflow?** Slides are Markdown sources in `slides/`; figures are referenced from `results/figures/`; numbers come from the CSV tables.
13. **What honesty caveats must be on the final slide?** Reduced budgets, one-seed ablations, synthetic control demand, and the results that contradicted the hypothesis.
