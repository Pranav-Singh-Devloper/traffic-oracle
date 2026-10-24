"""Residual hybrid: ML base forecast + graph residual.

XGBoost handles local autoregression and calendar effects but has no neighbor information.
STGAT is trained to learn only what XGBoost misses, r = y - y_xgb, seeing y_xgb as an extra
decoder input. The last decoder layer is zero-initialised, so training starts exactly at the
XGBoost forecast. Residual targets must come from OUT-OF-FOLD base predictions (see
scripts/make_oof.py): in-sample XGBoost residuals are unrealistically small.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from traffic.models.stgat import STGAT


class ResidualHybrid(nn.Module):
    def __init__(self, stgat: STGAT):
        super().__init__()
        self.net = stgat
        last = stgat.decoder[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, base: torch.Tensor, **_) -> torch.Tensor:
        """x [B,H,N,C], base [B,F,N] (scaled) -> base + learned residual (scaled)."""
        return base + self.net(x, base=base)


def oof_folds(n_anchors: int, n_blocks: int = 5, gap: int = 23) -> list[tuple[int, slice]]:
    """Forward-chaining folds over chronological train anchors.

    Returns [(train_end, predicted_block)] for blocks 2..n_blocks: train on anchors
    ``[0, train_end)`` (the last ``gap`` anchors before the block are left out so no training
    target overlaps the predicted block's inputs) and predict ``predicted_block``. Block 1 gets
    no out-of-fold forecast.
    """
    edges = [int(e) for e in np.linspace(0, n_anchors, n_blocks + 1)]  # floor, as in make_oof
    return [(max(edges[k] - gap, 0), slice(edges[k], edges[k + 1])) for k in range(1, n_blocks)]
