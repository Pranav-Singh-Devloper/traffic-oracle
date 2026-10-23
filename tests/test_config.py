import pytest

from traffic.config import load_config
from traffic.seed import get_device, set_seed


def test_load_base():
    cfg = load_config("configs/base.yaml")
    assert cfg["history_steps"] == 12


def test_inherits_and_override():
    cfg = load_config("configs/dl/stgat.yaml", ["debug.epochs=3"])
    assert cfg["model"]["name"] == "stgat" and cfg["debug"]["epochs"] == 3


def test_bad_split_rejected():
    with pytest.raises(ValueError):
        load_config("configs/base.yaml", ["split.train=0.9"])


def test_seed_and_device():
    set_seed(1)
    assert get_device().type in {"cpu", "cuda", "mps"}
