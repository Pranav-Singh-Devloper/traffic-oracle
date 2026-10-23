"""No-graph deep-learning ablation: one GRU shared across sensors, no spatial exchange.

Same inputs, width, node embedding and decoder as STGAT, minus the graph layers, so any gap
between this model and STGAT is attributable to the graph (not to "deep learning in general").
"""

from __future__ import annotations

import torch
from torch import nn


class GRUNoGraph(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        in_channels: int,
        horizon: int,
        d: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        node_embedding: bool = True,
    ):
        super().__init__()
        self.embed = nn.Linear(in_channels, d)
        self.node_emb = nn.Parameter(torch.randn(n_nodes, d) * 0.02) if node_embedding else None
        self.gru = nn.GRU(
            d, d, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.decoder = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, horizon)
        )

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        """x: [B, H, N, C] -> [B, F, N] (scaled units)."""
        b, h, n, _ = x.shape
        z = self.embed(x)  # [B, H, N, d]
        if self.node_emb is not None:
            z = z + self.node_emb[None, None]
        z = z.permute(0, 2, 1, 3).reshape(b * n, h, -1)  # each node is its own sequence
        out, _ = self.gru(z)
        y = self.decoder(out[:, -1])  # [B*N, F]
        return y.view(b, n, -1).permute(0, 2, 1)
