import numpy as np

from traffic.control.alinea import Alinea
from traffic.control.ctm import CTM
from traffic.control.predictive_alinea import occupancy_from_speed


def _run(ctm, demand, ramp_demand=0.0, steps=1500, rate=1800.0):
    rd = {c: ramp_demand for c in ctm.onramps}
    rr = {c: rate for c in ctm.onramps}
    for _ in range(steps):
        ctm.step(demand, rd, rr)


def test_vehicle_conservation():
    c = CTM()
    rd = {k: 500.0 for k in c.onramps}
    rr = {k: 1800.0 for k in c.onramps}
    for _ in range(2000):
        c.step(5000.0, rd, rr)
    demanded = 5000.0 * c.dt * 2000 + sum(500.0 * c.dt * 2000 for _ in c.onramps)
    assert abs(demanded - c.total_vehicles() - c.out_total) < 1e-6


def test_free_flow_at_low_demand():
    c = CTM()
    _run(c, 1500.0, 100.0)
    assert (c.speed() > 64).all() and c.upstream_queue < 1.0


def test_queue_forms_and_moves_backward():
    c = CTM()
    rd = {k: 400.0 for k in c.onramps}
    rr = {k: 1800.0 for k in c.onramps}
    fronts = []
    for k in range(1500):
        c.step(5200.0, rd, rr)  # above the bottleneck capacity of cell 10
        if k % 100 == 99:
            cong = np.flatnonzero(c.speed() < 40)
            fronts.append(cong.min() if len(cong) else c.n_cells)
    assert fronts[-1] < fronts[0] or fronts[0] < 10  # the congested region grows upstream
    assert (
        fronts[-1] <= fronts[3] and fronts[-1] < 9
    )  # congestion reached upstream of the bottleneck


def test_alinea_direction_and_clipping():
    a = Alinea(o_target=15.0)
    assert a.update(30.0) < 1800.0  # too much occupancy -> reduce rate
    for _ in range(100):
        a.update(60.0)
    assert a.rate == a.r_min
    for _ in range(100):
        a.update(0.0)
    assert a.rate == a.r_max


def test_occupancy_from_speed():
    from traffic.control.ctm import FDParams

    fd = FDParams()
    assert occupancy_from_speed(64.0, fd) is None
    assert occupancy_from_speed(20.0, fd) > occupancy_from_speed(45.0, fd) > 0
