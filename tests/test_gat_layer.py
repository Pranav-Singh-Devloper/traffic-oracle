"""Our dense GAT must match torch_geometric's GATConv / GATv2Conv numerically."""

import pytest
import torch

pyg = pytest.importorskip("torch_geometric.nn")

from traffic.models.layers.gat_dense import DenseGATLayer  # noqa: E402


def _graph(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    adj = (torch.rand(n, n, generator=g) < 0.25).float()
    adj.fill_diagonal_(0)
    # adj[i, j] = 1 <=> i aggregates from j <=> PyG edge (source j -> target i)
    edge_index = adj.nonzero().flip(1).T.contiguous()
    return adj, edge_index


def _check(concat: bool, v2: bool):
    n, fin, fout, k = 12, 8, 4, 3
    adj, ei = _graph(n)
    layer = DenseGATLayer(fin, fout, heads=k, concat=concat, v2=v2).eval()
    if v2:
        ref = pyg.GATv2Conv(
            fin, fout, heads=k, concat=concat, add_self_loops=True, share_weights=False
        ).eval()
    else:
        ref = pyg.GATConv(fin, fout, heads=k, concat=concat, add_self_loops=True).eval()
    with torch.no_grad():
        if v2:
            layer.lin_src.weight.copy_(ref.lin_l.weight)
            layer.lin_src.bias.copy_(ref.lin_l.bias)
            layer.lin_dst.weight.copy_(ref.lin_r.weight)
            layer.lin_dst.bias.copy_(ref.lin_r.bias)
            layer.att.copy_(ref.att)
        else:
            layer.lin_src.weight.copy_(ref.lin.weight)
            layer.att_src.copy_(ref.att_src)
            layer.att_dst.copy_(ref.att_dst)
        layer.bias.copy_(torch.randn_like(layer.bias))
        ref.bias.copy_(layer.bias)
        x = torch.randn(1, n, fin)
        ours, alpha = layer(x, adj, return_attn=True)
        theirs, (ei_a, a_ref) = ref(x[0], ei, return_attention_weights=True)
    torch.testing.assert_close(ours[0], theirs, atol=1e-5, rtol=1e-5)
    # compare attention weights edge by edge (PyG returns edges incl. self loops: [src, dst])
    dense = torch.zeros(k, n, n)
    dense[:, ei_a[1], ei_a[0]] = a_ref.T
    torch.testing.assert_close(alpha[0], dense, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("concat", [True, False])
def test_gat_v1_matches_pyg(concat):
    _check(concat, v2=False)


def test_gatv2_matches_pyg():
    _check(True, v2=True)


def test_attention_rows_sum_to_one_and_mask_non_neighbors():
    adj, _ = _graph()
    layer = DenseGATLayer(8, 4, heads=2).eval()
    _, a = layer(torch.randn(2, 12, 8), adj, return_attn=True)
    torch.testing.assert_close(a.sum(-1), torch.ones(2, 2, 12))
    nbr = (adj > 0) | torch.eye(12, dtype=torch.bool)
    assert (a[:, :, ~nbr] == 0).all()


def test_mean_attention_and_edge_bias_run():
    adj, _ = _graph()
    x = torch.randn(2, 12, 8)
    for kw in ({"attention": "mean"}, {"edge_bias": True}):
        out = DenseGATLayer(8, 4, heads=2, **kw)(x, adj)
        assert out.shape == (2, 12, 8)
