# Control experiment: assumptions

> This is an offline simulation on a synthetic network. It demonstrates how forecasts *could* inform a control policy. No live or real-world traffic control is performed or claimed.

1. **Network:** 12 cells x 0.5 mi, 3 lanes, 10 s steps (CFL: 65 mph x 10 s = 0.18 mi < 0.5 mi). Three on-ramps (cells 2, 5, 8); a lane-drop-like bottleneck at cell 10 with 75% capacity.
2. **Fundamental diagram:** triangular, v_free 65 mph, capacity 2000 veh/h/lane, jam density 200 veh/mi/lane (backward wave about 12 mph). Congested cells discharge at 88% of capacity (capacity drop 12%). Merges give the mainline 70% priority when capacity-limited.
3. **Demand is synthetic.** METR-LA has speed only, and inverting speed to flow is ambiguous in free flow (all sub-critical densities share one speed). Demand is therefore shaped by a *congestion index* ci(t) = mean over the 12 cell sensors of (65 - v)/45, divided by its **training-period** 95th percentile and clipped to [0, 1]. Mainline demand = 6000 x (0.45 + 0.22 ci) veh/h; each ramp = 250 + 150 ci veh/h. At ci = 1 total demand is about 16% above the bottleneck capacity.
4. **Sensor-to-cell map:** 12 evenly spaced sensors along the corridor found in Phase 9.
5. **Days:** the 5 test weekdays with the most AM congestion on the corridor; simulated 04:30-11:00, scored 05:00-11:00.
6. **Forecasts:** STGAT seed 0 test predictions (+15 min speed at the cell sensors), held for each 5-minute slot. The simulator is **open loop**: its own congestion never feeds back into the forecast model.
7. **Policies:** no control; fixed-time (rate tuned on one validation-period day, 06:00-10:00); ALINEA (K_R = 70 veh/h per % occupancy, target = critical occupancy); *predictive ALINEA* (literal spec: forecast speed -> density on the congested FD branch, fed back as max(measured, forecast-implied)); *look-ahead ALINEA* (a copy of the simulator is rolled 15 min ahead from the live state with demand moving to the forecast congestion index, and ALINEA is fed the max of measured and look-ahead occupancy); oracle versions of both use the true future speeds.
8. **Metrics:** total time spent (vehicles in cells + ramp queues + upstream queue), distance, mainline mean speed, maximum ramp queue (no spillback onto surface streets is modelled), congested cell-minutes (< 40 mph).
9. **Known limits:** no capacity-drop calibration to data; no ramp-queue override; the demand model is a modelling choice, so absolute numbers are illustrative only.
