import numpy as np
import pandas as pd
import pytest

from traffic.data.graph import (
    build_adj_from_distances,
    get_adjacency,
    graph_stats,
    verify_against_pkl,
)
from traffic.data.load import load_all


@pytest.mark.real_data
def test_rebuild_matches_pkl():
    m = load_all("data/raw")
    w = build_adj_from_distances(m.distances, m.sensor_ids)
    assert verify_against_pkl(w, m.adj)["ok"]


def test_direction_convention_chain():
    """Road runs a -> b -> c. W[i, j] comes from distance i -> j (row aggregates from column)."""
    d = pd.DataFrame(
        {
            "from": ["a", "b", "a", "a", "b", "c", "b", "c", "c"],
            "to": ["a", "b", "b", "c", "c", "c", "a", "a", "b"],
            "cost": [0, 0, 100, 200, 100, 0, 1e6, 1e6, 1e6],
        }
    )
    w = build_adj_from_distances(d, ["a", "b", "c"], 0.1)
    assert w[0, 1] > 0 and w[1, 0] == 0  # a aggregates from downstream b
    assert get_adjacency("backward", w)[1, 0] > 0  # b aggregates from upstream a


def test_variants(synthetic_graph):
    _, w = synthetic_graph
    n = w.shape[0]
    off = ~np.eye(n, dtype=bool)
    assert get_adjacency("bidirectional", w).shape == (n, n)
    ident = get_adjacency("identity", w)
    assert ((ident > 0) & off).sum() == 0
    rnd = get_adjacency("random", w)
    assert ((rnd > 0) & off).sum() == ((w > 0) & off).sum()
    assert graph_stats(w)["nodes"] == n
    with pytest.raises(ValueError):
        get_adjacency("nope", w)
