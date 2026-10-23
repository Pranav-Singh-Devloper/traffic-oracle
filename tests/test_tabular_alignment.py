import pytest

from traffic.config import load_config
from traffic.features.build_tabular import build_tabular, targets


@pytest.mark.real_data
def test_row_order_matches_targets():
    from traffic.data.pipeline import prepare

    cfg = load_config("configs/base.yaml")
    p = prepare(cfg, debug=True)
    tab = build_tabular(p, "test", cfg, "global", spatial=True)
    a, n = tab.X.shape[:2]
    assert tab.flat().shape[0] == a * n
    y, _ = targets(p, "test", 5)
    flat = y.ravel()
    # row r = anchor a_i * N + sensor j maps to speed[anchor + 5, j]
    r = 7 * n + 3
    assert flat[r] == p.speed[tab.anchors[7] + 5, 3]
    # lag_0 of that row is the imputed speed at the anchor for that sensor
    assert tab.flat()[r, tab.names.index("lag_0")] == pytest.approx(p.x_in[tab.anchors[7], 3])
    assert tab.flat()[r, tab.names.index("sensor_idx")] == 3
