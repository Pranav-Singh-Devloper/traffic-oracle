import numpy as np
import torch

from traffic.data.graph import get_adjacency
from traffic.models.gru_nograph import GRUNoGraph
from traffic.models.stgat import STGAT


def _adj(n=10, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.random((n, n)) * (rng.random((n, n)) < 0.3)
    np.fill_diagonal(w, 1.0)
    return torch.as_tensor(w)


def test_shapes_and_attention():
    adj = _adj()
    for kw in (
        {},
        {"temporal": "gru"},
        {"readout": "attn"},
        {"attention": "gatv2"},
        {"attention": "mean"},
        {"edge_bias": True},
        {"layers": 3, "dropedge": 0.2},
    ):
        m = STGAT(adj, in_channels=8, horizon=12, d=16, heads=4, **kw)
        y, attn = m(torch.randn(3, 12, 10, 8), return_attn=True)
        assert y.shape == (3, 12, 10)
        assert attn.shape == (3, 12, 4, 10, 10)
    g = GRUNoGraph(10, 8, 12, d=16)
    assert g(torch.randn(3, 12, 10, 8)).shape == (3, 12, 10)


def test_identity_graph_has_no_cross_node_flow():
    """With an identity graph node i's output must not depend on node j's inputs."""
    n = 6
    adj = torch.as_tensor(get_adjacency("identity", np.ones((n, n))))
    m = STGAT(adj, 8, 12, d=16, heads=4).eval()
    x = torch.randn(2, 12, n, 8)
    y1 = m(x)
    x2 = x.clone()
    x2[:, :, 3] += 5.0
    y2 = m(x2)
    keep = [i for i in range(n) if i != 3]
    torch.testing.assert_close(y1[:, :, keep], y2[:, :, keep])
    assert (y1[:, :, 3] - y2[:, :, 3]).abs().max() > 1e-4


def test_graph_model_does_use_neighbors():
    adj = _adj()
    adj = (adj + adj.T).clamp(max=1.0)
    m = STGAT(adj, 8, 12, d=16, heads=4).eval()
    x = torch.randn(2, 12, 10, 8)
    x2 = x.clone()
    x2[:, :, 0] += 5.0
    nb = [j for j in range(1, 10) if adj[j, 0] > 0][0]
    assert (m(x)[:, :, nb] - m(x2)[:, :, nb]).abs().max() > 1e-5


def test_temporal_conv_is_causal():
    m = STGAT(_adj(), 8, 12, d=16, heads=4, layers=2).eval()
    x = torch.randn(1, 12, 10, 8)
    z = m.blocks[0].temporal(m.embed(x))
    x2 = x.clone()
    x2[:, 8:] += 3.0  # change only the last 4 steps
    z2 = m.blocks[0].temporal(m.embed(x2))
    torch.testing.assert_close(z[:, :8], z2[:, :8])
