# Graph Attention Networks (GAT)

Implemented in `src/traffic/models/layers/gat_dense.py` and tested against PyG (`tests/test_gat_layer.py`).

## Equations (one head)
- `h'_i = W h_i`
- `e_ij = LeakyReLU(aᵀ [h'_i ‖ h'_j])` for `j ∈ N(i) ∪ {i}`, `−∞` otherwise
- `α_ij = softmax_j(e_ij)`  (masked softmax over neighbors)
- `h_i^out = Σ_j α_ij h'_j (+ b)`, then ELU outside the layer. K heads are concatenated (averaged in an output layer).
Splitting `a = [a_dst; a_src]` gives `e_ij = a_dstᵀ h'_i + a_srcᵀ h'_j`, which is what we compute (cheap: two dot products, then a broadcast sum).

## Why dense and masked
N = 207, so an [N x N] attention matrix per head is small; masking with the adjacency (non-edges to −∞) gives exactly the sparse softmax and batches trivially. Cost O(B·K·N²·d) vs sparse O(B·K·|E|·d).

## Direction convention
`adj[i, j] > 0` means node i aggregates from node j (ADR-002). PyG's `edge_index` is (source j, target i) so the test builds it with `adj.nonzero().flip(1).T`.

## GCN vs GAT vs GATv2
- **GCN** weights neighbors by fixed normalised adjacency (`attention="mean"` ablation): same weights for every input.
- **GAT** learns input-dependent weights, but its attention is *static* in ranking: with `aᵀ[Wh_i‖Wh_j]` the ordering of neighbors j is the same for every query node i (the score splits into a function of i plus a function of j, so softmax over j ignores i).
- **GATv2** moves the nonlinearity inside: `aᵀ LeakyReLU(W[h_i‖h_j])` ⇒ dynamic attention (`gatv2` ablation). We compute it in checkpointed chunks because the [B,N,N,K,D] tensor is large.

## Edge bias
`e_ij += β·W_ij` adds a learnable distance prior (ablation A12).

## Likely viva questions
1. *What does the mask do?* Restricts the softmax to neighbors so non-neighbors get exactly 0 attention.
2. *Why LeakyReLU?* Keeps gradients for negative scores; ELU is applied to the output.
3. *Why multiple heads?* Different heads can specialise (e.g. upstream vs downstream), and averaging stabilises training.
4. *What is "static attention"?* See above; GATv2 fixes it.
5. *Complexity?* See above.
6. *Why add self-loops?* So a node can keep its own information; also avoids all −∞ rows.
