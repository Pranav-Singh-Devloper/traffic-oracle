"""Build adjacency from distances; compare with adj_mx.pkl; graph variants and stats.

Direction convention (ADR-002): ``A[i, j] > 0`` means node i AGGREGATES FROM node j,
and ``W[i, j]`` (DCRNN) is built from the road distance from sensor i to sensor j.
So in ``W`` node i receives from the sensors it drives toward (downstream); in ``W.T``
node i receives from sensors that drive into it (upstream).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_adj_from_distances(
    dist_df: pd.DataFrame, sensor_ids: list[str], normalized_k: float = 0.1
) -> np.ndarray:
    """Thresholded Gaussian kernel exactly as DCRNN's ``gen_adj_mx``.

    sigma is the std of the finite entries of the N x N distance matrix.
    """
    n = len(sensor_ids)
    d = np.full((n, n), np.inf)
    idx = {s: i for i, s in enumerate(sensor_ids)}
    for f, t, c in dist_df[["from", "to", "cost"]].itertuples(index=False):
        if f in idx and t in idx:
            d[idx[f], idx[t]] = c
    sigma = d[~np.isinf(d)].flatten().std()
    w = np.exp(-np.square(d / sigma))
    w[w < normalized_k] = 0
    return w


def verify_against_pkl(w_rebuilt: np.ndarray, w_pkl: np.ndarray, tol: float = 1e-6) -> dict:
    """Report the difference between the rebuilt matrix and the canonical pickle."""
    diff = np.abs(w_rebuilt - w_pkl)
    return {
        "max_abs_diff": float(diff.max()),
        "n_differing": int((diff > tol).sum()),
        "ok": bool(diff.max() < tol),
    }


def _haversine(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    dla, dlo = la[:, None] - la[None], lo[:, None] - lo[None]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None] * np.sin(dlo / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def get_adjacency(
    variant: str,
    w: np.ndarray,
    locations: pd.DataFrame | None = None,
    k: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Graph variants used in ablations (see SPEC section 5.6)."""
    n = w.shape[0]
    if variant == "forward":
        return w.copy()
    if variant == "backward":
        return w.T.copy()
    if variant == "bidirectional":
        return w + w.T
    if variant == "identity":
        return np.eye(n)
    if variant == "random":
        rng = np.random.default_rng(seed)
        off = ~np.eye(n, dtype=bool)
        n_edges = int(((w > 0) & off).sum())
        weights = rng.permutation(w[(w > 0) & off])
        out = np.zeros((n, n))
        flat = rng.choice(np.flatnonzero(off.ravel()), size=n_edges, replace=False)
        out.ravel()[flat] = weights
        np.fill_diagonal(out, np.diag(w))
        return out
    if variant == "knn_geo":
        assert locations is not None, "knn_geo needs sensor locations"
        d = _haversine(locations["latitude"].to_numpy(), locations["longitude"].to_numpy())
        np.fill_diagonal(d, np.inf)
        sigma = d[np.isfinite(d)].std()
        out = np.zeros((n, n))
        for i in range(n):
            nn = np.argsort(d[i])[:k]
            out[i, nn] = np.exp(-((d[i, nn] / sigma) ** 2))
        np.fill_diagonal(out, 1.0)
        return out
    raise ValueError(f"unknown graph variant: {variant}")


def graph_stats(w: np.ndarray) -> dict:
    """NetworkX summary of the directed graph (off-diagonal non-zeros are edges)."""
    import networkx as nx

    a = (w > 0) & ~np.eye(w.shape[0], dtype=bool)
    g = nx.from_numpy_array(a.astype(int), create_using=nx.DiGraph)
    wcc = list(nx.weakly_connected_components(g))
    scc = list(nx.strongly_connected_components(g))
    big_scc = max(scc, key=len)
    sub = g.subgraph(big_scc)
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "density": nx.density(g),
        "self_loops": int((np.diag(w) > 0).sum()),
        "in_degree_hist": np.bincount([d for _, d in g.in_degree()]).tolist(),
        "out_degree_hist": np.bincount([d for _, d in g.out_degree()]).tolist(),
        "n_weak_components": len(wcc),
        "n_strong_components": len(scc),
        "largest_wcc": max(len(c) for c in wcc),
        "largest_scc": len(big_scc),
        "diameter_largest_scc": nx.diameter(sub) if len(big_scc) > 1 else 0,
    }
