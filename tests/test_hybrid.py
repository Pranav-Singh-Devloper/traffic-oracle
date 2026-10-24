import numpy as np
import torch

from traffic.config import load_config
from traffic.models.factory import build_model
from traffic.models.hybrid import oof_folds


def test_hybrid_starts_exactly_at_base_forecast():
    cfg = load_config("configs/hybrid/residual_hybrid.yaml")
    w = np.eye(10)
    w[0, 1] = 1.0
    m = build_model(cfg, w, None, 10).eval()
    x, base = torch.randn(2, 12, 10, 8), torch.randn(2, 12, 10)
    assert torch.equal(m(x, base), base)


def test_oof_folds_have_no_overlap_and_skip_block_one():
    n, gap = 23967, 23
    folds = oof_folds(n, 5, gap)
    assert len(folds) == 4 and folds[0][1].start > 0  # block 1 never predicted
    covered = np.zeros(n, bool)
    for train_end, blk in folds:
        # the last training anchor's targets (a + F) must precede the block's first input
        assert train_end + gap <= blk.start
        assert train_end > 0
        covered[blk] = True
    assert covered.sum() == n - folds[0][1].start  # blocks 2..5 are fully covered
    assert not covered[: folds[0][1].start].any()


def test_gate_mixture_is_convex():
    import sys
    from importlib import import_module

    sys.path.insert(0, "scripts")
    th = import_module("train_hybrid")
    gate = torch.nn.Sequential(torch.nn.Linear(16, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    gi = np.random.rand(5, 12, 16).astype(np.float32)
    a, b = np.random.rand(5, 12, 3), np.random.rand(5, 12, 3) + 10
    mixed, w = th.mix_forecasts(gate, gi, a, b)
    assert (mixed >= torch.as_tensor(a, dtype=torch.float32) - 1e-5).all() and (
        mixed <= torch.as_tensor(b, dtype=torch.float32) + 1e-5
    ).all()
    assert ((w > 0) & (w < 1)).all()
