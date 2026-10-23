"""Masked losses (torch). Missing targets contribute nothing.

Dividing by the number of observed entries (not the total) keeps the loss scale
independent of how many values are missing in a batch.
"""

from __future__ import annotations

import torch


def _norm(m: torch.Tensor) -> torch.Tensor:
    return m.sum().clamp(min=1.0)


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(pred.dtype)
    return ((pred - target).abs() * m).sum() / _norm(m)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(pred.dtype)
    return (((pred - target) ** 2) * m).sum() / _norm(m)


def masked_huber_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    m = mask.to(pred.dtype)
    e = (pred - target).abs()
    h = torch.where(e <= delta, 0.5 * e**2, delta * (e - 0.5 * delta))
    return (h * m).sum() / _norm(m)


LOSSES = {"mae": masked_mae_loss, "mse": masked_mse_loss, "huber": masked_huber_loss}
