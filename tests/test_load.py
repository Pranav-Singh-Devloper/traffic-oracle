import numpy as np
import pandas as pd
import pytest

from traffic.data.load import check_speed, load_all, load_speed


@pytest.mark.real_data
def test_real_load_and_alignment():
    m = load_all("data/raw")
    assert m.speed.shape == (34272, 207)
    assert list(m.speed.columns) == m.sensor_ids
    assert (m.locations["sensor_id"].tolist()) == m.sensor_ids
    assert m.speed.to_numpy().max() <= 100


def test_check_speed_rejects_bad_shape(synthetic_metr):
    with pytest.raises(AssertionError):
        check_speed(synthetic_metr)


def test_check_speed_rejects_gap():
    idx = pd.date_range("2012-03-01", periods=34272, freq="5min").delete(10)
    idx = idx.append(pd.DatetimeIndex(["2012-06-28"]))
    df = pd.DataFrame(np.ones((34272, 207)), index=idx)
    with pytest.raises(AssertionError):
        check_speed(df)


@pytest.mark.real_data
def test_read_hdf_fallback_matches_columns():
    assert all(isinstance(c, str) for c in load_speed("data/raw/metr-la.h5").columns)
