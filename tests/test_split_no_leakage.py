import pytest

from traffic.config import load_config
from traffic.data.split import chronological_split, dcrnn_sample_split
from traffic.data.windows import build_channels, make_windows


def test_strict_windows_do_not_cross_boundaries(synthetic_metr):
    idx = synthetic_metr.index
    sp = chronological_split(len(idx))
    y = synthetic_metr.to_numpy()
    chan = build_channels(y, y > 0, idx)
    w = {
        k: make_windows(chan, y, y > 0, idx, 12, 12, s.start, s.stop)
        for k, s in sp.as_dict().items()
    }
    assert w["train"].target_ts().max() < w["val"].input_ts().min()
    assert w["val"].target_ts().max() < w["test"].input_ts().min()


def test_split_sizes():
    sp = chronological_split(1000)
    assert (sp.train.stop, sp.val.stop, sp.test.stop) == (700, 800, 1000)


def test_dcrnn_split_numbers():
    sp = dcrnn_sample_split(34272 - 11 - 12)
    assert (
        sp.train.stop - sp.train.start,
        sp.val.stop - sp.val.start,
        sp.test.stop - sp.test.start,
    ) == (23974, 3425, 6850)


@pytest.mark.real_data
def test_real_strict_split():
    from traffic.data.pipeline import prepare

    p = prepare(load_config("configs/base.yaml"))
    w = p.windows
    assert w["train"].target_ts().max() < w["val"].input_ts().min()
    assert w["val"].target_ts().max() < w["test"].input_ts().min()
