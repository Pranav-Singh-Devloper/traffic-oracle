"""ALINEA local feedback ramp metering (Papageorgiou et al. 1991). Simulation only.

r(k) = r(k-1) + K_R * (o_target - o(k)), clipped to [r_min, r_max]; o is the occupancy (%)
measured downstream of the on-ramp merge. K_R = 70 veh/h per percent occupancy is the usual value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Alinea:
    o_target: float  # target occupancy in %, typically the critical occupancy
    k_r: float = 70.0
    r_min: float = 200.0
    r_max: float = 1800.0
    rate: float = 1800.0  # current metering rate (veh/h)

    def update(self, occupancy_pct: float) -> float:
        self.rate = min(
            self.r_max, max(self.r_min, self.rate + self.k_r * (self.o_target - occupancy_pct))
        )
        return self.rate
