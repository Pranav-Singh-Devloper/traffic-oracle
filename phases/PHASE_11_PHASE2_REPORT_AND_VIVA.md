# Phase 11 — Phase 2 Report, Slides, Viva Prep (M11)

**Goal:** finish the deliverables: Phase 2 IEEE report, slides, theory notes, per-member viva prep, and a reproducible README.
**SPEC refs:** §15, §16, §17 (Phase 2 structure), §18, §19, §20, §14 M11.
**Prerequisites:** Phases 6–9 (10 if done).
**Owner:** Vaibhav (assembly, docs); each member writes their own viva file and theory notes

---

## 11.1 Final evaluation pass

1. `make phase2` (or rerun `scripts/evaluate_all.py`) so every CSV is current.
2. Review `sanity_warnings.txt`. Any MAE@60 < 3.0 or any model losing to persistence at 15 min must be explained before writing.
3. `scripts/make_report_assets.py` regenerates `report/tables/*.tex`, `numbers.tex`, figures, and the README results block.

## 11.2 `report/phase2/main.tex` (IEEEtran)

Sections (SPEC §17):
1. Abstract
2. Introduction — hypothesis recap, Phase 1 gap result, contributions
3. DL Literature Review — DCRNN, STGCN, Graph WaveNet, ASTGCN, GMAN, GAT, GATv2, GCN
4. Data, Graph & Regularization — adjacency construction + direction convention, missing data, dropout / weight decay / early stopping / attention dropout / DropEdge / augmentation, each justified
5. Architecture — STGAT (diagram), dense GAT equations, GRU-nograph, DCRNN-lite (if done)
6. Hybrid Method & Rationale — residual learning, OOF forward-chaining (diagram of folds), gating
7. Experimental Setup — strict protocol, H = 12 for all, tuning budgets, 3 seeds, DM + bootstrap; literature-comparable (DCRNN protocol) numbers clearly labeled if reported
8. Results — main table, significance table, error-vs-horizon key figure, peak/off-peak
9. Ablations — table + bars, one paragraph per question
10. Interpretability & Congestion Maps — space-time diagram, error map, attention (qualitative)
11. Simulated Control — **with disclaimer**; results table
12. Limitations & Ethics — single city/season, speed only, fixed distance graph, open-loop simulation gap, MAPE at low speeds; ethics: surveillance concerns of sensor data are low (aggregated loop detectors), misuse risk of overclaiming control
13. Conclusion
14. References

All numbers through `numbers.tex` macros. `latexmk -pdf` must compile cleanly.

## 11.3 Slides — `slides/phase2.md`

~15 slides: recap → hypothesis → STGAT architecture → GAT equations (one slide) → no-graph ablation → main results → key error-vs-horizon figure → significance → ablations (A1/A2/A3/A4 highlight) → hybrid + where gains come from → space-time diagram → attention map → control (disclaimer on slide) → limitations → conclusion. Speaker notes as HTML comments.

## 11.4 Theory notes — `docs/theory/` (SPEC §15)

Write the remaining files (Phase 5 wrote 1, 2, 3, 8):
`gat.md`, `diffusion_conv.md`, `spatiotemporal_modeling.md`, `missing_data.md`, `significance_testing.md`, `hybrid.md`, `control.md`.
Each: equations, intuition, a worked small example, and 5–10 "likely viva question → answer" pairs. Link to the exact source file implementing each idea.

## 11.5 Viva prep — `docs/viva/<member>.md` (SPEC §16)

- One file per member (names from `CONTRIBUTIONS.md`).
- ~20 questions on their own modules + the 10 shared questions below, each with a short answer and a pointer to code.
- Shared questions (everyone must answer):
  1. Why a chronological split and not random? What leaks otherwise?
  2. Strict vs DCRNN protocol — what's the difference?
  3. Why is speed 0 treated as missing? How do masked losses handle it?
  4. How is the adjacency built? What are σ and κ? Which direction does information flow?
  5. Write the GAT attention equations. What does the mask do?
  6. How does GAT differ from GCN? What does GATv2 fix?
  7. Why does the hybrid need out-of-fold predictions?
  8. What does the GRU-nograph ablation prove that XGBoost vs STGAT doesn't?
  9. What is the Diebold–Mariano test and why HAC?
  10. What are the headline results and the biggest limitation?

## 11.6 README (final)

- One-paragraph summary, key figure, results table (auto-inserted between markers).
- One-command reproduction: `make all`; per-step commands; `--debug` quickstart.
- Expected runtimes and hardware used (read from `docs/experiment_log.md`, not guessed).
- Data provenance (DCRNN link) and the control disclaimer.

## 11.7 Definition of Done check (SPEC §20)

Write `docs/definition_of_done.md` ticking each SPEC §20 item with the file that proves it. Anything not done is listed as such, with the reason.

## 11.8 Final repo hygiene

- `CONTRIBUTIONS.md` complete (real names, modules, experiments, report sections).
- `ruff` + `black` clean, CI green.
- Large artifacts git-ignored; `make all` regenerates them. Optional: upload final checkpoints to a GitHub release.
- `refs.bib` entries all verified.

---

## Acceptance (M11)

- [ ] Phase 2 report compiles; every number from generated files
- [ ] Phase 2 slides complete
- [ ] All 11 theory notes and all member viva files exist
- [ ] README reproducible with results table + key figure
- [ ] `docs/definition_of_done.md` shows SPEC §20 status honestly
- [ ] CI green

**Commit:** `M11: Phase 2 report, slides, theory notes, viva prep`
**STOP — project complete pending team review.**
