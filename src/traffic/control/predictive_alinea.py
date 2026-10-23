"""Forecast-informed ALINEA. Simulation only.

Same control law as ALINEA, but the occupancy fed back is the larger of the measured one and
the occupancy implied by a 15-minute-ahead SPEED forecast of the downstream sensor. The forecast
speed is converted to density with the congested branch of the triangular fundamental diagram,
rho = w * rho_jam / (v + w); in free flow a speed carries no density information, so the
measured value is used. Result: the controller starts metering before the breakdown arrives.
"""

from __future__ import annotations

from dataclasses import dataclass

from traffic.control.alinea import Alinea
from traffic.control.ctm import FDParams


def occupancy_from_speed(v: float, fd: FDParams, eps: float = 3.0) -> float | None:
    """Occupancy (%) implied by a congested-branch speed; None if (near) free flow."""
    if v >= fd.v_free - eps:
        return None
    rho = fd.w * fd.rho_jam / (v + fd.w)  # veh/mi/lane
    return 100.0 * rho / fd.rho_jam


@dataclass
class PredictiveAlinea(Alinea):
    fd: FDParams = None  # type: ignore[assignment]

    def update_with_forecast(self, occupancy_pct: float, forecast_speed: float) -> float:
        o_hat = occupancy_from_speed(forecast_speed, self.fd)
        o = occupancy_pct if o_hat is None else max(occupancy_pct, o_hat)
        return self.update(o)
