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


class BatchIterator:
    """Vectorised batches straight from the window arrays (much faster than a DataLoader).

    Shuffling is only used for the train split; val/test keep chronological order.
    Yields (X [B,H,N,C], Y [B,F,N] mph, M [B,F,N] bool, anchor positions).
    """

    def __init__(
        self,
        w: Windows,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        channels: list[int] | None = None,
    ):
        self.w, self.bs, self.shuffle, self.channels = w, batch_size, shuffle, channels
        self.rng = np.random.default_rng(seed)
        self.hoff = np.arange(-w.H + 1, 1)
        self.foff = np.arange(1, w.F + 1)

    def __len__(self) -> int:
        return -(-len(self.w) // self.bs)

    def __iter__(self):
        n = len(self.w)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for i in range(0, n, self.bs):
            pos = order[i : i + self.bs]
            a = self.w.anchors[pos]
            x = self.w.chan[a[:, None] + self.hoff]
            if self.channels is not None:
                x = x[..., self.channels]
            yield (
                torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(self.w.y[a[:, None] + self.foff].astype(np.float32)),
                torch.from_numpy(self.w.ymask[a[:, None] + self.foff]),
                pos,
            )
