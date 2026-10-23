"""Dense masked Graph Attention layer (explainable, numerically equal to PyG's GATConv).

Direction convention (ADR-002): ``adj[i, j] > 0`` means node i AGGREGATES FROM node j.

For one head, with h'_i = W h_i:
    e_ij   = LeakyReLU(a_dst . h'_i + a_src . h'_j)      (= a^T [h'_i || h'_j], a = [a_dst; a_src])
    alpha  = softmax_j(e_ij) over j in N(i) U {i}  (non-neighbors are masked to -inf)
    out_i  = sum_j alpha_ij h'_j (+ bias)
Heads are concatenated (or averaged). ELU is applied by the caller so that this layer is
directly comparable with ``torch_geometric.nn.GATConv``.

Variants (ablation A4): ``v2`` = GATv2, e_ij = a . LeakyReLU(W_dst h_i + W_src h_j); and
``attention='mean'`` = fixed, row-normalised adjacency weights (GCN-style, no learned attention).
``edge_bias`` adds beta * W_ij to the score (distance prior, ablation A12).

Cost is O(B K N^2 d), fine for N = 207; a sparse version would be O(B K |E| d).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DenseGATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        concat: bool = True,
        attn_dropout: float = 0.0,
        negative_slope: float = 0.2,
        edge_bias: bool = False,
        v2: bool = False,
        attention: str = "gat",
    ):
        super().__init__()
        self.in_dim, self.out_dim, self.heads, self.concat = in_dim, out_dim, heads, concat
        self.v2, self.attention, self.slope = v2, attention, negative_slope
        self.attn_dropout = nn.Dropout(attn_dropout)
        hd = heads * out_dim
        # `lin_src` makes the messages (and, for GAT v1, also the target-side projection).
        self.lin_src = nn.Linear(in_dim, hd, bias=v2)
        self.lin_dst = nn.Linear(in_dim, hd, bias=True) if v2 else None
        if v2:
            self.att = nn.Parameter(torch.empty(1, heads, out_dim))
        else:
            self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
            self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(hd if concat else out_dim))
        self.beta = nn.Parameter(torch.zeros(heads)) if edge_bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_src.weight)
        if self.lin_dst is not None:
            nn.init.xavier_uniform_(self.lin_dst.weight)
        for name in ("att", "att_src", "att_dst"):
            if hasattr(self, name):
                nn.init.xavier_uniform_(getattr(self, name))

    def _v2_scores(self, xd: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """GATv2 scores [B, K, N, N]. The [B,N,N,K,D] intermediate is huge, so compute it in
        gradient-checkpointed chunks of graphs to keep memory bounded."""
        from torch.utils.checkpoint import checkpoint

        def chunk(xd_c, xs_c):
            s = xd_c.unsqueeze(2) + xs_c.unsqueeze(1)  # [b, N(i), N(j), K, D]
            return (F.leaky_relu(s, self.slope) * self.att).sum(-1).permute(0, 3, 1, 2)

        parts = []
        for i in range(0, xd.shape[0], 32):
            if self.training and torch.is_grad_enabled():
                parts.append(checkpoint(chunk, xd[i : i + 32], xs[i : i + 32], use_reentrant=False))
            else:
                parts.append(chunk(xd[i : i + 32], xs[i : i + 32]))
        return torch.cat(parts, 0)

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """h: [B, N, in_dim], adj: [N, N] -> [B, N, heads*out_dim] (or out_dim), attn [B,K,N,N]."""
        b, n, _ = h.shape
        k, d = self.heads, self.out_dim
        eye = torch.eye(n, dtype=torch.bool, device=h.device)
        nbr = (adj > 0) | eye  # self-loops are always present
        xs = self.lin_src(h).view(b, n, k, d)  # messages from j
        if self.attention == "mean":
            w = torch.where(
                nbr, adj.clamp(min=0) + eye.to(adj.dtype) * (adj == 0), adj.new_zeros(1)
            )
            alpha = (w / w.sum(-1, keepdim=True).clamp(min=1e-12))[None, None].expand(b, k, n, n)
        else:
            if self.v2:
                xd = self.lin_dst(h).view(b, n, k, d)
                e = self._v2_scores(xd, xs)
            else:
                a_src = (xs * self.att_src).sum(-1)  # [B, N, K]
                a_dst = (xs * self.att_dst).sum(-1)
                e = F.leaky_relu(
                    a_dst.permute(0, 2, 1)[..., :, None] + a_src.permute(0, 2, 1)[..., None, :],
                    self.slope,
                )
            if self.beta is not None:
                e = e + self.beta.view(1, k, 1, 1) * adj.clamp(min=0)[None, None]
            e = e.masked_fill(~nbr[None, None], float("-inf"))
            alpha = torch.softmax(e, dim=-1)
        alpha_d = self.attn_dropout(alpha)
        out = torch.einsum("bkij,bjkd->bikd", alpha_d, xs)
        out = out.reshape(b, n, k * d) if self.concat else out.mean(2)
        out = out + self.bias
        return (out, alpha) if return_attn else out
