# Viva prep: Pranav Singh (rank 1): deep-learning core (GAT layer, STGAT, GRU, trainer)

Team T042. Also answer `shared_questions.md` (10 questions everyone must know). Ownership: see `CONTRIBUTIONS.md`.

1. **Describe the STGAT architecture.** Embed → +node embedding → L blocks (gated causal TCN → multi-head dense GAT per time step → ELU/dropout → residual+LayerNorm) → last-step readout → MLP decoder predicting all 12 horizons.
2. **Why dense attention?** N=207 is small; masking by adjacency gives the exact sparse softmax and batches easily.
3. **How did you prove your GAT is correct?** Numerical equivalence with PyG GATConv/GATv2Conv (outputs and attention) in `test_gat_layer.py`.
4. **How do you map our adjacency to PyG's edge_index?** adj[i,j]=1 ⇔ edge j→i; edge_index = adj.nonzero().flip(1).T.
5. **What is the self-loop handling?** Self-loops always allowed, so rows never become all −∞.
6. **What is a gated TCN and why causal padding?** Conv with GLU gate; left padding ensures step t only sees ≤ t (tested).
7. **Why a node embedding?** Gives each sensor identity; ablation A7.
8. **What is the loss and why in scaled space?** Masked MAE on scaled targets (stable), early stopping on val MAE in mph.
9. **What regularisation is used?** Dropout, attention dropout, weight decay, early stopping, optional DropEdge/noise/sensor masking.
10. **How does early stopping work?** Patience on val MAE (mph), best checkpoint restored.
11. **Why 3 seeds?** To report mean ± std and avoid seed luck.
12. **Why is the GRU ablation parameter-matched?** To isolate the graph, not capacity.
13. **What does the identity-graph test show?** Node i's output is independent of node j's input; confirms no leakage across nodes.
14. **What is GATv2's memory issue and fix?** [B,N,N,K,D] tensor is huge; compute in checkpointed chunks.
15. **How do you keep GPU/CPU load laptop-friendly?** Small model, batch 64, nice priority, capped threads, sequential runs.
16. **How did you time the model and decide on the compute simplification?** Profiled iterations on MPS: dense attention at all 12 steps cost 3.7 min/epoch; attention is memory-bound. Applying GAT on the last 4 steps and training on a third of the windows per epoch gives about 0.6 min/epoch (ADR-007).
17. **What is the sparse GAT layer and why is it exact?** It stores neighbour lists padded to D_max=26 and computes the same softmax over each node's neighbourhood; tested numerically equal to the dense layer and PyG. It did not speed things up because gathers dominate.
18. **How does the trainer keep validation consistent with the metric?** Training uses masked MAE in scaled space; validation inverse-transforms to mph and early-stops on masked MAE in mph.
19. **Why MPS instead of CPU, and what are the risks?** MPS was about 7x faster than CPU for STGAT; risks are op coverage and nondeterminism, so we fix seeds and report 3 seeds.
20. **What did the three-seed comparison with the GRU show?** Equal at 15 min, slightly better at 30 (not significant), clearly better at 60 min (DM significant); see Table of main results.

## Verification log
- 2026-10-24 13:26: Pranav Singh ran the tests for their modules (tests/test_gat_layer.py, tests/test_stgat_shapes.py): 10 passed in 2.76s.
