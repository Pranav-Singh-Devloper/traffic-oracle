# Diffusion convolution (DCRNN)

**Owner:** Vaibhav

Optional model (`dcrnn_lite`); the main project uses attention instead.

## Idea
Traffic on a directed road graph behaves like a random walk. With transition matrices `D_O⁻¹ W` (forward) and `D_I⁻¹ Wᵀ` (backward), a K-step bidirectional diffusion filter is
`Z = Σ_{k=0}^{K−1} ( θ_{k,1} (D_O⁻¹W)^k + θ_{k,2} (D_I⁻¹Wᵀ)^k ) X`.
Inside a GRU cell, the matrix multiplications of the gates are replaced by diffusion convolutions, giving the DCRNN encoder–decoder.

## Why both directions
Information flows downstream (vehicles move) but congestion waves travel upstream, so the model needs both `W` and `Wᵀ`.

## Contrast with GAT
Diffusion weights are fixed by distance; GAT learns weights from node features. Diffusion reaches K hops through matrix powers, while stacked GAT layers reach L hops.

## Likely viva questions
1. *Why is the graph directed?* Road distance from i to j differs from j to i.
2. *What does K control?* Receptive field in hops.
3. *Why a random walk?* Gives a normalised, stochastic operator whose powers stay bounded.
