"""Torch Dataset for windows; shuffling is only ever done inside the train split."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from traffic.data.windows import Windows


class TrafficWindowDataset(Dataset):
    """Returns (X [H,N,C], Y [F,N] mph, mask [F,N], window index)."""

    def __init__(self, w: Windows):
        self.w = w

    def __len__(self) -> int:
        return len(self.w)

    def __getitem__(self, i: int):
        y, m = self.w.target(i)
        return (
            torch.from_numpy(np.ascontiguousarray(self.w.x(i))),
            torch.from_numpy(y.astype(np.float32)),
            torch.from_numpy(m),
            i,
        )


def make_loaders(
    windows: dict[str, Windows], batch_size: int = 64, num_workers: int = 0
) -> dict[str, DataLoader]:
    return {
        k: DataLoader(
            TrafficWindowDataset(w),
            batch_size=batch_size,
            shuffle=(k == "train"),
            num_workers=num_workers,
        )
        for k, w in windows.items()
    }
