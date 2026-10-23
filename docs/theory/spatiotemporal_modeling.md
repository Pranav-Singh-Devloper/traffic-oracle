# Spatio-temporal modeling

## Shockwaves
Congestion starts at a bottleneck and the jam's tail moves upstream at roughly 10–15 mph (our CTM backward wave speed w ≈ 12 mph). A sensor upstream of a bottleneck will slow down minutes after downstream sensors do. A per-segment model sees only its own past and learns this only indirectly, while a graph model can read the downstream state directly.

## Receptive field
Spatial reach = L (GAT layers) hops; temporal reach = H input steps (TCN receptive field grows with dilation: kernel 3, dilations 1, 2, 4...). A shockwave at speed w covers distance w·Δt in time Δt, so the graph depth needed for horizon h grows with h. Our evidence: correlation decays with graph distance (EDA fig. 6), but directional lags are only weakly asymmetric (EDA fig. 7).

## Over-smoothing
Repeated neighborhood averaging makes node representations similar. Residual connections, LayerNorm and few layers (L = 2 default, tested at 1 and 3, ablation A6) limit this.

## Non-autoregressive decoding
All 12 horizons are predicted jointly: no error accumulation from feeding predictions back, at the cost of no explicit horizon-to-horizon dependency.

## Likely viva questions
1. *Why a gated causal temporal conv?* Parallel over time (faster than RNN), causal padding avoids using future steps (tested).
2. *Why a node embedding?* Gives each sensor an identity (local behaviour); ablation A7.
3. *What would more layers do?* Larger receptive field but over-smoothing and cost.
