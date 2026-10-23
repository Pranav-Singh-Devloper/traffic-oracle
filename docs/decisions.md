# Architecture Decision Records

Template: **Context / Decision / Consequences / Status**

## ADR-001: Repository root
- **Context:** `DL_AML_Project/` is nested inside the parent git repo `folder_2/`. CI and per-member history need their own repo.
- **Decision:** Recommended: `git init` here and push to a new GitHub repo.
- **Consequences:** Pending the owner's go-ahead; nothing has been initialised.
- **Status:** Proposed.

## ADR-002: Graph message direction
- **Context:** DCRNN's `W[i, j]` is built from the road distance from sensor i to sensor j. The spec needs one explicit convention.
- **Decision:** `A[i, j] > 0` means node i aggregates from node j (row = receiver). `forward = W`: i receives from sensors it drives toward (downstream, where shockwaves originate). `backward = W.T`: i receives from sensors that drive into it (upstream). Checked on a 3-sensor chain in `tests/test_graph.py`.
- **Consequences:** The GAT layer, spatial features and variants all use this convention. `adj_mx.pkl` stays canonical; the rebuild from distances matches it (max abs diff 1.0e-7, float32 rounding).
- **Status:** Accepted.

## ADR-003: Scaling of missing inputs
- **Context:** Scaling a raw 0 gives about -4.6 sigma, a fake "very slow" value.
- **Decision:** Scale only observed/imputed values. Under `zero_mask`, missing positions are set to 0.0 in scaled space (the train mean) and the `is_missing` channel carries the outage signal. Under `ffill_tod_mean` (default) they are forward-filled up to 6 steps, then filled with the train-only time-of-day mean.
- **Status:** Accepted.

## ADR-004: Reading metr-la.h5
- **Context:** `pd.read_hdf` raises `TypeError: a bytes-like object is required, not 'str'` on this 2019 file with pandas 3.1 / PyTables 3.11.
- **Decision:** `load_speed` tries `pd.read_hdf` first and falls back to reading the plain fixed-format layout (`axis0`, `axis1` int64 ns, `block0_values`) with PyTables. Same data and index; asserted by the integrity checks.
- **Status:** Accepted.
