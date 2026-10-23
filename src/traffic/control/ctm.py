"""Cell Transmission Model (Daganzo 1994) for a freeway corridor with on-ramps.

OFFLINE SIMULATION ONLY: this is a synthetic network that shows how forecasts *could* inform a
control policy. No live or real-world traffic control is performed or claimed.

Units: miles, hours, vehicles. Triangular fundamental diagram per lane: free-flow speed v_free,
capacity q_max, jam density rho_jam, backward wave speed w = q_max / (rho_jam - q_max / v_free).
Per step and cell: sending S_i = min(v_free*dt/L * n_i, Q_i*dt), receiving
R_i = min(Q_i*dt, w*dt/L * (N_jam_i - n_i)); flow i->i+1 = min(S_i, R_{i+1}). At an on-ramp
cell the mainline and the ramp share R_i; if both cannot be served, the mainline gets a
priority share ``merge_priority`` (a documented modelling choice).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FDParams:
    v_free: float = 65.0  # mph
    q_max: float = 2000.0  # veh/h/lane
    rho_jam: float = 200.0  # veh/mi/lane
    lanes: int = 3

    @property
    def rho_crit(self) -> float:
        return self.q_max / self.v_free

    @property
    def w(self) -> float:
        return self.q_max / (self.rho_jam - self.rho_crit)


@dataclass
class CTM:
    n_cells: int = 12
    cell_len: float = 0.5  # miles
    dt: float = 10.0 / 3600  # hours
    fd: FDParams = field(default_factory=FDParams)
    onramps: tuple[int, ...] = (2, 5, 8)  # cells receiving a ramp
    ramp_capacity: float = 1800.0  # veh/h, max ramp discharge when unmetered
    merge_priority: float = 0.7  # mainline share when merging is capacity-limited
    bottleneck: dict[int, float] = field(
        default_factory=lambda: {10: 0.75}
    )  # cell -> capacity factor

    def __post_init__(self) -> None:
        # CFL: no vehicle may cross more than one cell per step
        assert self.fd.v_free * self.dt <= self.cell_len, "CFL violated"
        fd, n = self.fd, self.n_cells
        cap = np.full(n, fd.q_max * fd.lanes)
        for c, f in self.bottleneck.items():
            cap[c] *= f
        self.cap = cap  # veh/h per cell
        self.n_jam = np.full(n, fd.rho_jam * fd.lanes * self.cell_len)
        self.reset()

    def reset(self) -> None:
        self.n = np.zeros(self.n_cells)  # vehicles per cell
        self.queue = {c: 0.0 for c in self.onramps}  # ramp queues (veh)
        self.upstream_queue = 0.0
        self.out_total = self.in_total = 0.0

    # --- derived quantities -------------------------------------------------
    def density(self) -> np.ndarray:  # veh/mi (all lanes)
        return self.n / self.cell_len

    def occupancy(self) -> np.ndarray:
        """Occupancy proxy in percent: density / jam density."""
        return 100.0 * self.n / self.n_jam

    def speed(self) -> np.ndarray:
        """Cell speed (mph) from the triangular FD, free-flow when under critical density."""
        rho = self.density() / self.fd.lanes
        v = np.full(self.n_cells, self.fd.v_free)
        cong = rho > self.fd.rho_crit
        v[cong] = np.minimum(self.fd.v_free, self.fd.w * (self.fd.rho_jam / rho[cong] - 1.0))
        return v

    # --- dynamics ------------------------------------------------------------
    def step(
        self, upstream_demand: float, ramp_demand: dict[int, float], ramp_rate: dict[int, float]
    ):
        """Advance one step. Demands in veh/h; ramp_rate = metering rate (veh/h) per ramp cell."""
        fd, dt, n = self.fd, self.dt, self.n_cells
        send = np.minimum(fd.v_free * dt / self.cell_len * self.n, self.cap * dt)
        recv = np.minimum(self.cap * dt, fd.w * dt / self.cell_len * (self.n_jam - self.n))
        inflow = np.zeros(n)
        outflow = np.zeros(n)
        # upstream boundary with a virtual queue
        want = self.upstream_queue + upstream_demand * dt
        s_up = min(want, self.cap[0] * dt)
        ramp_send = {
            c: min(self.queue[c] + ramp_demand[c] * dt, min(ramp_rate[c], self.ramp_capacity) * dt)
            for c in self.onramps
        }
        for i in range(n):
            s_main = s_up if i == 0 else send[i - 1]
            s_r = ramp_send.get(i, 0.0)
            r = recv[i]
            if s_main + s_r <= r:
                f_main, f_ramp = s_main, s_r
            else:  # capacity-limited merge
                f_main = min(s_main, max(self.merge_priority * r, r - s_r))
                f_ramp = min(s_r, r - f_main)
            inflow[i] = f_main + f_ramp
            if i == 0:
                self.upstream_queue = want - f_main
                self.in_total += f_main
            else:
                outflow[i - 1] = f_main
            if i in self.onramps:
                self.queue[i] += ramp_demand[i] * dt - f_ramp
                self.in_total += f_ramp
        exit_flow = min(send[-1], self.cap[-1] * dt)
        outflow[-1] = exit_flow
        self.out_total += exit_flow
        self.n = self.n + inflow - outflow
        return {"flow": outflow / dt, "speed": self.speed(), "occupancy": self.occupancy()}

    def total_vehicles(self) -> float:
        """Vehicles anywhere in the system: cells + ramp queues + upstream virtual queue."""
        return float(self.n.sum() + sum(self.queue.values()) + self.upstream_queue)
