"""Residual hybrid: ML base forecast + graph residual.

XGBoost handles local autoregression and calendar effects but has no neighbor information.
STGAT is trained to learn only what XGBoost misses, r = y - y_xgb, seeing y_xgb as an extra
decoder input. The last decoder layer is zero-initialised, so training starts exactly at the
XGBoost forecast. Residual targets must come from OUT-OF-FOLD base predictions (see
scripts/make_oof.py): in-sample XGBoost residuals are unrealistically small.
"""

from __future__ import annotations

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
