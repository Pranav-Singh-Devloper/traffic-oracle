# Simulated control: CTM, ALINEA

**Disclaimer:** offline simulation on a synthetic network; no live or real-world control is performed or claimed.

## Cell Transmission Model
Discretises a freeway into cells of length L; vehicles per cell n_i. Per step: sending `S_i = min(v·dt/L · n_i, Q_i·dt)`, receiving `R_i = min(Q_i·dt, w·dt/L·(N_jam − n_i))`, flow `min(S_i, R_{i+1})`. Triangular fundamental diagram: free-flow speed v, capacity q_max at critical density ρ_c = q_max/v, backward wave speed `w = q_max / (ρ_jam − ρ_c)`. CFL: `v·dt ≤ L`. Merges share receiving capacity with a mainline priority.

## ALINEA
`r(k) = r(k−1) + K_R (o* − o(k))`: an integral controller steering downstream occupancy to the critical value. Reactive: it responds after density rises.

## Predictive ALINEA
Replace o(k) by `max(o_measured, o_from_forecast)`, where the forecast is the STGAT +15 min speed converted to density on the congested FD branch. In free flow, speed is uninformative about density on a triangular FD, so the measured value is used. Oracle version uses true future speed (upper bound).

## Honest limits
METR-LA has speed only, so demand is a synthetic profile shaped by the observed congestion index. The simulator is open loop: its congestion does not influence the forecasts. Real ramp metering also involves safety, queue-override and driver behaviour not modelled. METR-LA freeways have no signals, hence ramp metering rather than intersection control.

## Likely viva questions
1. *Why CTM?* Simple, conservative, reproduces queue growth and backward waves.
2. *Why ALINEA?* Standard, robust feedback law with one gain.
3. *Why can speed not give density in free flow?* All sub-critical densities have the same speed on a triangular FD.
