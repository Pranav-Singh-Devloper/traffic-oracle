# Viva prep: Pranav Singh (rank 1): deep-learning core and hybrid models

Also answer `shared_questions.md` (10 questions everyone must know). Ownership: see `CONTRIBUTIONS.md`.

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
14. **What is over-smoothing and how do you test it?** Representations of neighbors converge with depth; ablation A6 (1/2/3 layers).
15. **What is GATv2's memory issue and fix?** [B,N,N,K,D] tensor is huge; compute in checkpointed chunks.
16. **How is the ablation budget reduced and why is it still fair?** Short epochs, one seed, with a same-budget default run as the reference.
17. **What does ablation A2 (random graph) test?** Whether road topology matters or just extra parameters.
18. **How are H=6/24 compared fairly?** On the intersection of test target timestamps.
19. **Why bidirectional adjacency as default?** More information and DCRNN also uses both directions; A3 tests directions separately.
20. **How do you keep GPU/CPU load laptop-friendly?** Small model, batch 64, nice priority, capped threads, sequential runs.
21. **How is the OOF procedure done?** 5 chronological train blocks; block k predicted by a model trained on earlier blocks with a gap; block 1 dropped.
22. **Why zero-initialise the residual head?** Hybrid starts as XGBoost; training can only add improvements.
23. **How is the gate trained and what is the caveat?** First 80% of val train, last 20% early-stop; val no longer pure for model selection.
24. **What would it mean if the hybrid doesn't beat STGAT?** STGAT already captures local structure; report honestly.
