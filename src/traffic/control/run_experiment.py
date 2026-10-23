"""Simulated ramp-metering experiment on a METR-LA-shaped synthetic corridor (CTM).

DISCLAIMER: This is an offline simulation on a synthetic network. It demonstrates how forecasts
*could* inform a control policy. No live or real-world traffic control is performed or claimed.

Demand is NOT measured flow (METR-LA has speed only). It is a synthetic profile shaped by the
corridor's observed congestion index ci(t) = mean over corridor sensors of
clip((65 - v)/(65 - 20), 0, 1), then scaled to the CTM capacity (see docs/control_assumptions.md).
The simulator is open loop: its own congestion does not feed back into the STGAT inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from traffic.control.alinea import Alinea
from traffic.control.ctm import CTM
from traffic.control.predictive_alinea import PredictiveAlinea

DISCLAIMER = (
    "This is an offline simulation on a synthetic network. It demonstrates how forecasts could "
    "inform a control policy. No live or real-world traffic control is performed or claimed."
)
POLICIES = ["no_control", "fixed_time", "alinea", "predictive_alinea", "oracle_predictive_alinea"]
WARMUP_H, START_H, END_H = 4.5, 5.0, 11.0


@dataclass
class DayInputs:
    """Per-5-minute inputs for one day, slots from WARMUP_H to END_H."""

    ci: np.ndarray  # [T] congestion index in [0, 1]
    fc_pred: np.ndarray  # [T, n_cells] +15 min speed forecast made at this slot
    fc_true: np.ndarray  # [T, n_cells] true speed 15 min later
    date: str


def demand(ci: float, ctm: CTM) -> tuple[float, dict[int, float]]:
    """Synthetic mainline and ramp demands (veh/h) shaped by the congestion index."""
    q_total = ctm.fd.q_max * ctm.fd.lanes
    main = q_total * (0.50 + 0.30 * ci)
    ramp = {c: 600.0 * (0.6 + 0.8 * ci) for c in ctm.onramps}
    return main, ramp


def simulate(policy: str, day: DayInputs, fixed_rate: float = 900.0, ctl_steps: int = 6) -> dict:
    ctm = CTM()
    fd = ctm.fd
    o_target = 100.0 * fd.rho_crit / fd.rho_jam
    ctrl = {}
    for c in ctm.onramps:
        if policy == "alinea":
            ctrl[c] = Alinea(o_target)
        elif policy.endswith("predictive_alinea"):
            ctrl[c] = PredictiveAlinea(o_target, fd=fd)
    rate = {c: ctm.ramp_capacity for c in ctm.onramps}
    steps_per_slot = int(round(5 / 60 / ctm.dt))
    tts = dist = veh_h = cong_cell_min = 0.0
    max_q = 0.0
    dens, queues = [], []
    n_steps = int((END_H - WARMUP_H) / ctm.dt)
    for k in range(n_steps):
        t_h = WARMUP_H + k * ctm.dt
        slot = min(int((t_h - WARMUP_H) * 12), len(day.ci) - 1)
        main, ramp = demand(day.ci[slot], ctm)
        if policy == "fixed_time" and 6.0 <= t_h < 10.0:
            rate = dict.fromkeys(ctm.onramps, fixed_rate)
        elif policy == "fixed_time":
            rate = dict.fromkeys(ctm.onramps, ctm.ramp_capacity)
        elif ctrl and k % ctl_steps == 0:
            occ = ctm.occupancy()
            for c, ct in ctrl.items():
                down = min(c + 1, ctm.n_cells - 1)
                if policy == "alinea":
                    rate[c] = ct.update(occ[down])
                else:
                    f = (day.fc_true if policy.startswith("oracle") else day.fc_pred)[slot, down]
                    rate[c] = ct.update_with_forecast(occ[down], f)
        out = ctm.step(main, ramp, rate)
        if t_h >= START_H:
            tts += ctm.total_vehicles() * ctm.dt
            veh_h += ctm.n.sum() * ctm.dt
            dist += out["flow"].sum() * ctm.dt * ctm.cell_len
            cong_cell_min += (out["speed"] < 40).sum() * ctm.dt * 60
            max_q = max(max_q, max(ctm.queue.values()))
            if k % steps_per_slot == 0:
                dens.append(ctm.density().copy())
                queues.append([ctm.queue[c] for c in ctm.onramps])
    return {
        "TTS_veh_h": tts,
        "distance_veh_mi": dist,
        "mainline_avg_speed_mph": dist / max(veh_h, 1e-9),
        "max_ramp_queue_veh": max_q,
        "congestion_cell_minutes": cong_cell_min,
        "_density": np.array(dens),
        "_queues": np.array(queues),
    }


def run_policies(days: list[DayInputs], fixed_rate: float) -> pd.DataFrame:
    rows = []
    for d in days:
        for pol in POLICIES:
            r = simulate(pol, d, fixed_rate)
            rows.append(
                {
                    "date": d.date,
                    "policy": pol,
                    **{k: v for k, v in r.items() if not k.startswith("_")},
                }
            )
    return pd.DataFrame(rows)


def tune_fixed_rate(
    val_day: DayInputs, grid=(500, 650, 800, 950, 1100, 1300)
) -> tuple[float, pd.DataFrame]:
    """Pick the fixed-time rate that minimises TTS on a VALIDATION-period day."""
    res = [(r, simulate("fixed_time", val_day, r)["TTS_veh_h"]) for r in grid]
    df = pd.DataFrame(res, columns=["rate_veh_h", "TTS_veh_h"])
    return float(df.loc[df.TTS_veh_h.idxmin(), "rate_veh_h"]), df
