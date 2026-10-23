# Phase 8 — Ablation Studies (M8)

**Goal:** change one thing at a time from the STGAT default, 3 seeds each, and answer a specific question per ablation.
**SPEC refs:** §11, §8.3 (H sweep), §9.2 (regularization), §14 M8.
**Prerequisites:** Phase 6 (and Phase 7 for A13).

---

## 8.1 Ablation configs — `configs/ablations/*.yaml`

Each file overrides **only** the keys that change; everything else inherits `configs/dl/stgat.yaml`.

| # | File(s) | Override | Question |
|---|---|---|---|
| A1 | `a01_identity.yaml` | `graph.variant: identity` | Does spatial info help at all? |
| A2 | `a02_random.yaml` | `graph.variant: random` (same edge count, seeded) | Road topology or just extra parameters? |
| A3 | `a03_{forward,backward,bidirectional}.yaml` | `graph.variant` | Does direction matter? |
| A4 | `a04_{gat,gatv2,mean}.yaml` | `model.attention` | Does attention beat fixed weights? |
| A5 | `a05_heads{1,2,4,8}.yaml` | `model.heads` (keep d = 64 total) | Multi-head benefit |
| A6 | `a06_layers{1,2,3}.yaml` | `model.layers` | Receptive field vs over-smoothing |
| A7 | `a07_no_node_emb.yaml` | `model.node_embedding: false` | Value of sensor identity |
| A8 | `a08_no_time.yaml` | drop tod/dow/weekend/holiday channels | Value of calendar context |
| A9 | `a09_zero_mask.yaml` | `input_imputation: zero_mask` | Missing-data handling |
| A10 | `a10_no_dropout.yaml`, `a10_no_wd.yaml` | `dropout: 0` / `weight_decay: 0` | Regularization effect |
| A11 | `a11_H{6,12,24}.yaml` | `history_steps` | Window sensitivity (**also for best ML model**) |
| A12 | `a12_edge_bias.yaml` | `model.edge_bias: true` | Distance prior in attention |
| A13 | — (reads Phase 7 outputs) | residual vs gating vs average | Hybrid design choice |

Optional extras if time allows (also one-change rules): `knn_geo` graph, per-sensor scaler, augmentation on/off, DropEdge.

Note: A9 and A11 change the data pipeline — the processed cache key must include `input_imputation` and `H`. A11 for H = 24 drops the first 12 anchors of each split; evaluate **all** H variants on the **intersection** of target timestamps so the comparison stays fair.

## 8.2 Runner — `scripts/run_ablations.py`

```
python scripts/run_ablations.py --ids A1 A2 ... --seeds 0 1 2 [--debug] [--resume]
```
- Loops configs × seeds, calls the same `train_dl.py` entry function (no duplicated training code).
- `--resume` skips runs whose `.npz` already exists with a matching config hash.
- Prints an estimated total runtime from the default STGAT epoch time before starting; asks for confirmation if > 12 h (or `--yes`).
- For A11 also runs the best ML model (from `main_table.csv`) with H ∈ {6, 12, 24}.

## 8.3 Outputs

- `results/metrics/ablations.csv`: `ablation, variant, seed, horizon, MAE, RMSE, MAPE` + aggregated mean ± std.
- `results/figures/ablations/ablation_bars.png`: grouped bars MAE@15/30/60 with ±std error bars, default STGAT as a reference line.
- `results/figures/ablations/h_sweep.png`: MAE vs H for STGAT and best ML.
- DM test for A1 (identity) and A2 (random) vs default at each headline horizon → `significance.csv`.
- `docs/ablation_analysis.md`: one paragraph per ablation answering its question, quoting numbers from the CSV.

## 8.4 Things to watch

- If A2 (random graph) ≈ default, the "road topology helps" claim is **not** supported — report it.
- If A1 ≈ default but STGAT ≫ GRU-nograph, check that the difference is not just node embeddings or capacity.
- Compute budget: 13 ablations × ~2–4 variants × 3 seeds ≈ 80–100 runs. If GPU time is limited, run the full set at reduced epochs first, then full runs for A1–A4 (the core claims). Document any reduction in an ADR.

---

## Acceptance (M8)

- [ ] All A1–A13 present in `ablations.csv` with 3 seeds (or reduction documented)
- [ ] Ablation bar chart + H-sweep figure
- [ ] `docs/ablation_analysis.md` written from CSV values

**Commit:** `M8: ablation studies`
**STOP for review.**
