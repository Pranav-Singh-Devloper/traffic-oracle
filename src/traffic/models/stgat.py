"""Spatio-temporal Graph Attention Network (main model).

    X [B,H,N,C] -> Linear C->d -> (+ node embedding)
      -> L x block: temporal (gated causal conv | GRU) -> spatial (multi-head dense GAT at every
                    time step) -> ELU -> dropout -> residual -> LayerNorm
      -> readout (last step | attention pooling over H) -> MLP decoder -> [B, F, N]

Non-autoregressive: all F horizons are predicted at once.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from traffic.models.layers.gat_dense import DenseGATLayer, SparseGATLayer


class GatedTemporalConv(nn.Module):
    """Causal dilated conv with a GLU gate, applied independently per node."""

    def __init__(self, d: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(d, 2 * d, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, N, d]
        b, t, n, d = x.shape
        z = x.permute(0, 2, 3, 1).reshape(b * n, d, t)
        z = F.glu(self.conv(F.pad(z, (self.pad, 0))), dim=1)
        return z.reshape(b, n, d, t).permute(0, 3, 1, 2)


class GRUTemporal(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, d = x.shape
        z = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
        return self.gru(z)[0].reshape(b, n, t, d).permute(0, 2, 1, 3)


class STBlock(nn.Module):
    def __init__(
        self,
        d,
        heads,
        dilation,
        temporal,
        dropout,
        attn_dropout,
        attention,
        edge_bias,
        v2,
        sparse_adj=None,
        spatial_steps: int = 0,
    ):
        super().__init__()
        self.spatial_steps = spatial_steps  # 0 = all steps; k = only the last k (cheaper)
        self.temporal = GRUTemporal(d) if temporal == "gru" else GatedTemporalConv(d, 3, dilation)
        if sparse_adj is not None:  # exact same function as the dense layer, ~8x less memory
            self.gat = SparseGATLayer(
                d, d // heads, sparse_adj, heads, True, attn_dropout, edge_bias=edge_bias
            )
        else:
            self.gat = DenseGATLayer(
                d,
                d // heads,
                heads,
                True,
                attn_dropout,
                edge_bias=edge_bias,
                v2=v2,
                attention=attention,
            )
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, return_attn: bool = False):
        b, t, n, d = x.shape
        z = self.temporal(x)
        k = t if self.spatial_steps in (0, None) or self.spatial_steps >= t else self.spatial_steps
        zs = z[:, t - k :]  # spatial attention on the last k steps only
        g = self.gat(zs.reshape(b * k, n, d), adj, return_attn=return_attn)
        attn = None
        if return_attn:
            g, attn = g
            attn = attn.view(b, k, *attn.shape[1:])  # [B, k, K, N, N]
        g = self.drop(F.elu(g)).view(b, k, n, d)
        if k < t:  # earlier steps get no spatial update (identity branch)
            g = torch.cat([torch.zeros(b, t - k, n, d, device=g.device, dtype=g.dtype), g], dim=1)
        out = self.norm(x + g)
        return (out, attn) if return_attn else out


class STGAT(nn.Module):
    def __init__(
        self,
        adj: torch.Tensor,
        in_channels: int,
        horizon: int,
        d: int = 64,
        heads: int = 4,
        layers: int = 2,
        temporal: str = "tcn",
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        node_embedding: bool = True,
        attention: str = "gat",  # gat | gatv2 | mean
        edge_bias: bool = False,
        readout: str = "last",
        dropedge: float = 0.0,
        base_horizon: int = 0,
        sparse: bool = True,
        spatial_steps: int = 4,
    ):
        super().__init__()
        n = adj.shape[0]
        self.register_buffer("adj", adj.float())
        self.dropedge, self.readout_kind = dropedge, readout
        self.embed = nn.Linear(in_channels, d)
        self.node_emb = nn.Parameter(torch.randn(n, d) * 0.02) if node_embedding else None
        self.blocks = nn.ModuleList(
            STBlock(
                d,
                heads,
                2**i,
                temporal,
                dropout,
                attn_dropout,
                "mean" if attention == "mean" else "gat",
                edge_bias,
                attention == "gatv2",
                adj.float() if (attention == "gat" and dropedge == 0.0 and sparse) else None,
                spatial_steps,
            )
            for i in range(layers)
        )
        self.pool = nn.Linear(d, 1) if readout == "attn" else None
        self.decoder = nn.Sequential(
            nn.Linear(d + base_horizon, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, horizon)
        )

    def _adj(self) -> torch.Tensor:
        if self.training and self.dropedge > 0:
            keep = (torch.rand_like(self.adj) > self.dropedge).float()
            keep.fill_diagonal_(1.0)
            return self.adj * keep
        return self.adj

    def forward(self, x: torch.Tensor, base: torch.Tensor | None = None, return_attn: bool = False):
        """x: [B, H, N, C] -> [B, F, N] in scaled units (and last-block attention if asked).

        ``base`` [B, F, N] (optional) is a scaled base forecast given to the decoder (hybrid).
        """
        z = self.embed(x)
        if self.node_emb is not None:
            z = z + self.node_emb[None, None]
        adj = self._adj()
        attn = None
        for i, blk in enumerate(self.blocks):
            if return_attn and i == len(self.blocks) - 1:
                z, attn = blk(z, adj, return_attn=True)
            else:
                z = blk(z, adj)
        if self.readout_kind == "attn":
            w = torch.softmax(self.pool(z).squeeze(-1), dim=1)  # [B, T, N]
            r = (w.unsqueeze(-1) * z).sum(1)
        else:
            r = z[:, -1]
        if base is not None:
            r = torch.cat([r, base.permute(0, 2, 1)], dim=-1)
        y = self.decoder(r).permute(0, 2, 1)  # [B, F, N]
        return (y, attn) if return_attn else y
