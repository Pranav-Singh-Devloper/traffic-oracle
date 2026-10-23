"""GAT attention summaries and maps (qualitative; attention is not causal evidence)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from traffic.data.datasets import BatchIterator


@torch.no_grad()
def mean_attention(
    model, windows, positions: np.ndarray, channels, device, batch_size: int = 32
) -> np.ndarray:
    """Average last-block attention at the last input step over given window positions: [N, N].

    Heads are averaged. Rows are receivers i, columns senders j (ADR-002).
    """
    from traffic.data.windows import Windows

    sub = Windows(
        windows.chan,
        windows.y,
        windows.ymask,
        windows.index,
        windows.anchors[positions],
        windows.H,
        windows.F,
    )
    model.eval()
    acc, n = None, 0
    for x, _, _, _ in BatchIterator(sub, batch_size, False, 0, channels):
        _, attn = model(x.to(device), return_attn=True)  # [B, T, K, N, N]
        a = attn[:, -1].mean(1).sum(0).cpu().numpy()
        acc = a if acc is None else acc + a
        n += x.shape[0]
    return acc / n


def attention_shares(att: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Attention mass share on self / downstream-only / upstream-only / both-direction links."""
    n = att.shape[0]
    eye = np.eye(n, dtype=bool)
    down = (w > 0) & ~(w.T > 0) & ~eye  # i receives from j that is downstream of i
    up = (w.T > 0) & ~(w > 0) & ~eye
    both = (w > 0) & (w.T > 0) & ~eye
    tot = att.sum()
    return {
        "self": att[eye].sum() / tot,
        "downstream_only": att[down].sum() / tot,
        "upstream_only": att[up].sum() / tot,
        "both_directions": att[both].sum() / tot,
    }


def plot_attention_map(
    att: np.ndarray,
    locations: pd.DataFrame,
    targets: list[int],
    title: str,
    out: Path,
    top_k: int = 6,
) -> None:
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5))
    for ax, i in zip(np.atleast_1d(axes), targets, strict=True):
        ax.scatter(locations.longitude, locations.latitude, s=8, c="lightgrey")
        row = att[i].copy()
        row[i] = 0
        for j in np.argsort(row)[::-1][:top_k]:
            ax.plot(
                [locations.longitude.iloc[j], locations.longitude.iloc[i]],
                [locations.latitude.iloc[j], locations.latitude.iloc[i]],
                color="C3",
                lw=1 + 25 * row[j],
                alpha=0.7,
            )
            ax.scatter(locations.longitude.iloc[j], locations.latitude.iloc[j], s=20, c="C0")
        ax.scatter(locations.longitude.iloc[i], locations.latitude.iloc[i], s=70, c="k", marker="*")
        la = locations.latitude.iloc[i]
        lo = locations.longitude.iloc[i]
        ax.set(
            xlim=(lo - 0.15, lo + 0.15), ylim=(la - 0.12, la + 0.12), title=f"{title}: sensor {i}"
        )
        ax.set_aspect(1 / np.cos(np.radians(la)))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
