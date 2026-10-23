# Phase 6 — Deep Learning Models (M6)

**Goal:** our own dense GAT layer (proven equal to PyG's), the no-graph GRU ablation, the main STGAT model, a reusable trainer, and 3-seed results. Optional DCRNN-lite.
**SPEC refs:** §9, §8.3, §14 M6.
**Prerequisites:** Phases 1 and 3 (Phase 4 for comparison).

---

## 6.1 Losses — `src/traffic/training/losses.py`

```python
def masked_mae_loss(pred, target, mask) -> Tensor      # Σ m|y−ŷ| / clamp(Σ m, min=1)
def masked_mse_loss(...); def masked_huber_loss(..., delta=1.0)
```
- Train in **scaled** space: scale `target` with the train scaler, keep `mask` from original `y > 0`.
- Test: torch losses match `evaluation/metrics.py` on the same arrays (after inverse scaling) to 1e-5.
- Note for the viva: dividing by Σm (not by total count) keeps the loss scale independent of how many values are missing.

## 6.2 Dense masked GAT — `src/traffic/models/layers/gat_dense.py`

```python
class DenseGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, concat=True, dropout=0.0, attn_dropout=0.1,
                 negative_slope=0.2, add_self_loops=True, edge_bias=False, v2=False): ...
    def forward(self, h: Tensor, adj: Tensor, return_attn=False) -> Tensor | (Tensor, Tensor)
    # h: [B, N, in_dim]; adj: [N, N] (row i aggregates from column j, per ADR-002)
    # out: [B, N, heads*out_dim] if concat else [B, N, out_dim]; attn: [B, heads, N, N]
```
Equations (GAT, Veličković 2018):
- `h'_i = W h_i` → split into K heads of size `out_dim`
- `e_ij = LeakyReLU(a_dstᵀ h'_i + a_srcᵀ h'_j)` (this equals `aᵀ[h'_i ‖ h'_j]` with `a = [a_dst; a_src]`)
- mask: `e_ij = −inf` where `adj[i, j] == 0` (after adding self-loops)
- `α_ij = softmax_j(e_ij)`, attention dropout on α
- `out_i = Σ_j α_ij h'_j` (+ bias); concat heads, or mean if `concat=False`
- ELU is applied **outside** the layer (in the block), so the layer matches PyG exactly.
- `edge_bias=True`: `e_ij += β · W_ij` with learnable scalar β per head (ablation A12).
- `v2=True` (GATv2, Brody 2022): `e_ij = aᵀ LeakyReLU(W_dst h_i + W_src h_j)` (ablation A4).
- Mean-aggregation mode (`attention="mean"`): α = row-normalized adjacency (GCN-style, A4).
- Rows with no neighbors other than self still work because of self-loops; guard against all −inf rows anyway (`nan_to_num`).
- Complexity O(B·K·N²·d) — fine for N = 207. State this vs sparse O(|E|·d) in the docstring.

### Equivalence test — `tests/test_gat_layer.py`
1. Random graph: N = 12, ~30 directed edges, seed fixed.
2. Build PyG `GATConv(in, out, heads=K, concat=True, add_self_loops=True, dropout=0)` and our layer with dropout 0.
3. Copy weights: `lin.weight` → our `W`; `att_src` → `a_src`; `att_dst` → `a_dst`; `bias` → bias. (Check names against the installed PyG version.)
4. **Edge direction:** PyG `edge_index[0]` = source j, `edge_index[1]` = target i. Our `adj[i, j] = 1` ⇔ edge `(j → i)`. Build `edge_index = adj.nonzero().flip(1).T`.
5. Assert outputs equal (atol 1e-5) and attention weights equal (via `return_attention_weights=True`).
6. Repeat for `concat=False` and for GATv2 vs `GATv2Conv` (share_weights=False).

## 6.3 `gru_nograph` — `src/traffic/models/gru_nograph.py` (mandatory ablation)

```
X [B, H, N, C] → reshape [B·N, H, C] → Linear C→d → (+ node embedding, same as STGAT) →
GRU(d, d, layers=2) → last hidden [B·N, d] → MLP d→d→F → [B, F, N]
```
- Same inputs, d, node embedding, dropout, and **roughly the same parameter count** as STGAT (report both counts). This isolates "graph" from "deep learning".

## 6.4 `stgat` — `src/traffic/models/stgat.py`

```
X [B, H, N, C]
 → Linear C→d (d=64)
 → + node embedding E[N, d] (toggle)
 → L=2 × STBlock:
      temporal: gated TCN  (tanh(conv) ⊙ σ(conv), kernel 3, dilation 1 then 2, causal padding)
                OR GRU per node (config: temporal: tcn|gru)
      spatial:  DenseGATLayer(d, d/K, heads=K=4, concat) applied at every time step
                (fold time into batch: [B·H, N, d])
      ELU → dropout → residual → LayerNorm
 → readout: last time step | attention pooling over H (config)
 → MLP decoder d → d → F   → [B, F, N]    (non-autoregressive)
```
- `forward(x, return_attn=False)` returns attention from the last block's last time step for visualization.
- Config keys cover every ablation in Phase 8 (graph variant, attention type, heads, layers, node embedding, time features, dropout, weight decay, edge bias, H). Ablations change config only — never code paths copied into new files.

## 6.5 `dcrnn_lite` (optional) — `src/traffic/models/dcrnn_lite.py` + `layers/diffusion.py`

- Diffusion conv: `Σ_k (θ_k,f (D_O⁻¹W)^k + θ_k,b (D_I⁻¹Wᵀ)^k) X`, K = 2, inside a GRU cell; encoder–decoder, 1–2 layers, d = 64.
- If time is short: wrap `torch_geometric_temporal.nn.recurrent.DCRNN` and record that in an ADR.
- Not required for acceptance.

## 6.6 Trainer — `src/traffic/training/trainer.py`

```python
class Trainer:
    def __init__(self, model, cfg, scaler, device)
    def fit(self, train_loader, val_loader) -> History
    def predict(self, loader) -> np.ndarray        # mph, [S, F, N]
```
- Adam lr 1e-3, weight decay 1e-4; `ReduceLROnPlateau(factor=0.5, patience=5)` or cosine (config).
- Grad clip 5.0; batch 64; max 100 epochs; early stopping on **val masked MAE (mph)** with patience 15; restore best checkpoint.
- AMP (`torch.autocast`) on CUDA only; MPS/CPU in fp32.
- Optional augmentation (train only, config): Gaussian input noise σ on scaled speed; random sensor masking (set speed→0-in-scaled-space and `is_missing`→1 for p% sensors per sample). DropEdge (randomly zero adjacency entries per batch) as a regularization option.
- Logs per epoch: train loss, val MAE, lr, time → `results/metrics/training/{run}.csv`; curves → `results/figures/training/{run}.png`.
- Checkpoints → `checkpoints/dl/{run}.pt` with config + scaler params + git hash.

`scripts/train_dl.py --model {gru_nograph|stgat|dcrnn_lite} --seeds 0 1 2 [--debug] [--set key=value ...]` → one `.npz` per seed.

## 6.7 Tests

- `test_gat_layer.py` (above).
- `test_stgat_shapes.py` (add): output `[B, F, N]`; attention rows sum to 1 over neighbors; masked (non-neighbor) attention entries are exactly 0.
- `test_identity_graph.py` (add): with `identity` adjacency, STGAT node outputs don't depend on other nodes' inputs (perturb node j, check node i unchanged).
- `test_end_to_end_debug.py`: every model (baselines, ML, DL) runs 1 epoch on the debug config in CI.

---

## Acceptance (M6)

- [ ] GAT equivalence test passes against PyG `GATConv` (and `GATv2Conv`)
- [ ] `gru_nograph` and `stgat` trained on 3 seeds; results in `main_table.csv` as mean ± std
- [ ] **STGAT beats persistence at all horizons** (SPEC M6)
- [ ] Training curves saved for every run
- [ ] DM tests: STGAT vs XGB-global and STGAT vs GRU-nograph in `significance.csv`
- [ ] Sanity warnings reviewed (MAE@60 < 3.0 → investigate leakage before celebrating)

**Commit:** `M6: dense GAT, GRU-nograph, STGAT, trainer, 3-seed results`
**STOP for review.**
