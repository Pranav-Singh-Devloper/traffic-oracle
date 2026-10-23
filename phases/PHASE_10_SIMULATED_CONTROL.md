# Phase 10 — Simulated Control (M10, optional)

> **Disclaimer (include in every file, figure caption, and slide of this phase):**
> "This is an offline simulation on a synthetic network. It demonstrates how forecasts *could* inform a control policy. No live or real-world traffic control is performed or claimed."

**Goal:** a pure-Python freeway corridor simulator (Cell Transmission Model) comparing ramp-metering policies, including one driven by STGAT forecasts.
**SPEC refs:** §12, §2.3, §14 M10.
**Prerequisites:** Phase 6 (STGAT predictions), Phase 9 (corridor sensor list).

---

## 10.1 CTM — `src/traffic/control/ctm.py`

Daganzo (1994) with a triangular fundamental diagram.

```python
@dataclass
class FDParams:
    v_free: float = 65.0      # mph
    q_max: float = 2000.0     # veh/h/lane (capacity)
    rho_jam: float = 200.0    # veh/mi/lane
    lanes: int = 4
    # derived: rho_crit = q_max / v_free; w (backward wave speed) = q_max / (rho_jam - rho_crit)

class CTM:
    def __init__(self, n_cells=12, cell_len_mi=0.5, dt_s=10.0, fd=FDParams(), onramps={3: ..., 7: ..., 10: ...}, offramps=...)
    def step(self, upstream_demand_vph, ramp_demand_vph: dict, ramp_rates_vph: dict) -> State
```
- **CFL condition:** `v_free · dt ≤ cell_len` → assert at init (65 mph × 10 s ≈ 0.18 mi ≤ 0.5 mi ✓).
- Sending `S_i = min(v ρ_i L, q_max)`, receiving `R_i = min(q_max, w(ρ_jam − ρ_i) L)`, flow `= min(S_{i−1}, R_i)`; merges at on-ramps with priority split (document the merge rule).
- Ramp queues: `queue += (demand − metered_flow)·dt`; cap with max queue (spillback penalty or overflow count).
- Downstream boundary: a bottleneck cell with reduced capacity (e.g. lane drop) so congestion actually forms.
- Tests (`tests/test_ctm.py`, add): vehicle conservation (in − out = Δstorage) to 1e-6; free-flow steady state with low demand; a queue forms upstream of the bottleneck with high demand and its front moves **backward**.

## 10.2 Demand from METR-LA — `control/run_experiment.py::build_demand`

1. Use the corridor from `results/metrics/corridor.json` (Phase 9); map sensors to cells.
2. Speed → density/flow via the triangular FD:
   - **Congested** (`v < v_free − ε`): invert the congested branch, `ρ = w·ρ_jam / (v + w)`, `q = ρ·v`.
   - **Free flow** (`v ≈ v_free`): speed carries no density information on a triangular FD (every ρ < ρ_crit has v = v_free). Use a time-of-day demand profile scaled to below capacity instead.
3. This inversion is approximate. Write every assumption in `docs/control_assumptions.md` and in the report.
4. Ramp demands: scaled copies of the mainline time-of-day profile (configurable ratio, default 10–15% of mainline).
5. Choose 5 test-set weekdays (programmatically: the 5 with the most AM-peak congestion on the corridor). Simulate 05:00–11:00.

## 10.3 Policies

| Policy | File | Rule |
|---|---|---|
| No control | `run_experiment.py` | `r = ∞` (ramp flow = demand, subject to merge) |
| Fixed-time | `run_experiment.py` | `r` = constant per ramp during 06:00–10:00 (tuned once on a val day) |
| ALINEA | `control/alinea.py` | `r(k) = r(k−1) + K_R·(o* − o(k))`, occupancy from downstream cell density (`o ≈ ρ/ρ_jam`), `K_R = 70 veh/h`, `o*` ≈ critical occupancy, clip `[r_min, r_max]` |
| Predictive ALINEA | `control/predictive_alinea.py` | Same law, but `o(k)` is replaced by occupancy inferred from the **STGAT +15 min forecast** of the downstream sensor speed (via FD inversion) |
| Oracle predictive | `control/predictive_alinea.py` | Same as predictive, using the **true** +15 min speed (upper bound) |

- Control interval 30 s–1 min; forecasts update every 5 min (hold between updates).
- Forecasts come from the saved STGAT test predictions (no model call inside the simulator) — keeps it deterministic and fast.
- Mismatch note: the simulator's own congestion does not feed back into STGAT inputs (open loop). State this limitation explicitly.

## 10.4 Metrics + outputs

- Total time spent (TTS, veh·h), total travel distance (veh·mi), mainline average speed, max ramp queue, congestion duration (cell-minutes below 40 mph).
- `results/metrics/control_results.csv`: policy × day × metric; summary mean ± std over 5 days.
- Figures: TTS bar chart per policy; density space-time heatmaps per policy for one day; ramp queue time series.
- `docs/control_results.md` with the disclaimer at the top.

## 10.5 SUMO alternative (only if the team asks)

`control/sumo/`: 3×3 grid, fixed-time / actuated / max-pressure / forecast-aware max-pressure via TraCI; metrics: delay, queue, throughput, stops. CityFlow is an acceptable lighter alternative. Skip unless requested.

---

## Acceptance (M10)

- [ ] CTM tests pass (conservation, backward-moving queue)
- [ ] All 5 policies run on 5 days; results table + figures
- [ ] Disclaimer present in every output of this phase
- [ ] Assumptions documented

**Commit:** `M10: simulated ramp-metering experiment (CTM)`
**STOP for review.**
