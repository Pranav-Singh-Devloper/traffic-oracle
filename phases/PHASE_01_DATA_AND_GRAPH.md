# Phase 1 — Data + Graph (M1)

**Goal:** download METR-LA, load it safely, rebuild and verify the road graph, build the missing mask, split by time, scale on train only, and build sliding windows. Everything later depends on this phase being leak-free.
**SPEC refs:** §2, §3, §5.2–5.7, §14 M1.
**Prerequisites:** Phase 0.
**Owner:** Kanchan

---

## 1.1 Download — `scripts/download_data.py`

Source: <https://github.com/liyaguang/DCRNN#data-preparation>

### Files and sources (try in order)

| Target in `data/raw/` | Source 1 (primary) | Fallbacks |
|---|---|---|
| `metr-la.h5` | Google Drive folder `10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX` via `gdown --folder` (or `gdown.download_folder(id=..., output=tmp)`; pick `metr-la.h5` from the result) | Kaggle mirror via `kaggle datasets download` if the user has a Kaggle token; otherwise manual instructions |
| `adj_mx.pkl` | `https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl` | — |
| `distances_la_2012.csv` | `.../data/sensor_graph/distances_la_2012.csv` | — |
| `graph_sensor_ids.txt` | `.../data/sensor_graph/graph_sensor_ids.txt` | — |
| `graph_sensor_locations.csv` | `.../data/sensor_graph/graph_sensor_locations.csv` | — |

Notes:
- Use `requests` with streaming, a 60 s timeout, and 3 retries with backoff. `distances_la_2012.csv` is the slowest file; a quick probe timed out once.
- Google Drive often rate-limits shared folders. If `gdown` fails, **do not** loop forever: print the manual instructions below and exit non-zero.
- PyTorch Geometric Temporal's `METRLADatasetLoader` ships `node_values.npy`/`adj_mat.npy` (already preprocessed, no DatetimeIndex). It **cannot** satisfy the integrity checks in 1.2, so do not use it as the source of truth. Mention it in `data/README.md` only.
- Download into a fresh temp dir, verify, then move into `data/raw/`. Never execute anything from downloaded files.
- Optional flag `--pems-bay` also fetches `pems-bay.h5` (+ `adj_mx_bay.pkl`, `graph_sensor_locations_bay.csv`) for the PeMS extension.

### Manual fallback message (print verbatim when auto-download fails)
```
Could not download METR-LA automatically.
1. Open https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX
   (or Baidu: https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) and download metr-la.h5
2. Download the sensor graph files from
   https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph
3. Place them at:
   data/raw/metr-la.h5
   data/raw/adj_mx.pkl
   data/raw/distances_la_2012.csv
   data/raw/graph_sensor_ids.txt
   data/raw/graph_sensor_locations.csv
4. Re-run: python scripts/download_data.py --verify-only
```

### Integrity + manifest
- `--verify-only` runs only the checks in 1.2.
- Write `data/raw/MANIFEST.json`: `{file: {bytes, sha256, source_url, downloaded_at}}`. Commit this file.

---

## 1.2 Loaders — `src/traffic/data/load.py`

```python
def load_speed(path) -> pd.DataFrame          # pd.read_hdf(path)  (key "df")
def load_adj_pkl(path) -> tuple[list[str], dict[str,int], np.ndarray]
    # pickle.load(f, encoding="latin1")  -> (sensor_ids, sensor_id_to_ind, adj_mx)
def load_distances(path) -> pd.DataFrame      # columns: from, to, cost
def load_locations(path) -> pd.DataFrame      # columns: index, sensor_id, latitude, longitude
def load_all(cfg) -> MetrLA                   # dataclass bundling the above, columns aligned
```

**Integrity asserts** (also in `tests/test_load.py`, marked `real_data`):
- `df.shape == (34272, 207)`; `34272 == 119 * 288`.
- `df.index[0] == Timestamp("2012-03-01 00:00:00")`, `df.index[-1] == Timestamp("2012-06-27 23:55:00")`.
- Index is strictly increasing with a constant 5-minute step (no gaps, no duplicates).
- No negative speeds; no NaN (zeros are the missing marker). Log the max speed (expect ≈70 mph).
- `adj_mx.shape == (207, 207)`, `len(sensor_ids) == 207`.

**ID alignment (common bug):**
- Normalize all sensor IDs to `str` (h5 columns, pkl ids, txt ids, csv ids can come as int or str).
- Assert `list(df.columns) == sensor_ids` from the pkl. If the order differs, **reorder the DataFrame columns to the pkl order** and log it. All arrays in the project use pkl order.
- `distances_la_2012.csv` contains more sensors than the 207; filter to the 207.
- Locations: assert all 207 sensors have lat/lon.

---

## 1.3 Graph — `src/traffic/data/graph.py`

### Rebuild from distances (DCRNN `gen_adj_mx` logic)
```python
def build_adj_from_distances(dist_df, sensor_ids, normalized_k=0.1) -> np.ndarray:
    N = len(sensor_ids); D = np.full((N, N), np.inf)
    idx = {s: i for i, s in enumerate(sensor_ids)}
    for f, t, c in dist_df.itertuples(index=False):
        if f in idx and t in idx: D[idx[f], idx[t]] = c
    sigma = D[~np.isinf(D)].flatten().std()
    W = np.exp(-np.square(D / sigma))
    W[W < normalized_k] = 0
    return W
```
- σ is the std of **finite** entries of the 207×207 matrix (this matches DCRNN exactly; it is *not* the std of all CSV rows).
- `verify_against_pkl(W_rebuilt, W_pkl)`: print max abs diff, number of differing entries, and assert `< 1e-6`. If it fails, write the differences to `results/metrics/adj_rebuild_diff.csv` and an ADR. **`adj_mx.pkl` stays canonical.**

### Direction convention (ADR-002, required)
In DCRNN's matrix, `W[i, j]` is built from the road distance **from sensor i to sensor j** (row = `from`, column = `to`).
For message passing we use the convention **"row i aggregates from column j"**. Decide and document which matrix gives which physical meaning:
- `forward` = `W` → node i aggregates from nodes it drives *toward* (downstream) — mostly downstream info flowing back to i. This matches shockwaves travelling upstream.
- `backward` = `Wᵀ` → node i aggregates from nodes that drive *into* it (upstream).
Write this in `docs/decisions.md` with a tiny 3-sensor example, and add a unit test that checks it on a hand-built chain graph.

### Graph variants
```python
def get_adjacency(variant, W, locations=None, k=8, seed=42) -> np.ndarray
# forward | backward | bidirectional (W + W.T) | identity (np.eye) |
# random (same number of non-zero off-diagonal edges, random positions, seeded, weights shuffled from W) |
# knn_geo (k nearest by haversine on lat/lon, Gaussian kernel weights)
```
All variants: add self-loops where the model needs them (the GAT layer handles this — don't double-add).

### NetworkX stats — `graph_stats(W) -> dict`
Build `nx.DiGraph` from non-zero off-diagonal entries. Report and save to `results/metrics/graph_stats.json`:
node count, edge count, density, in/out-degree distribution (save histogram), # weakly and strongly connected components, size of largest WCC/SCC, diameter of the largest SCC (unweighted), self-loop count.

---

## 1.4 Missing data — `src/traffic/data/missing.py`

```python
def build_mask(speed: np.ndarray) -> np.ndarray          # (speed > 0).astype(bool), [T, N]
def outage_stats(mask, index) -> dict                     # overall rate, per-sensor rate, longest runs,
                                                          # count of whole-network outage timestamps
def tod_means(speed, mask, index, train_idx) -> np.ndarray  # [288, N], train timestamps only, observed values only
def impute_inputs(speed, mask, index, tod_mean, max_ffill=6, method="ffill_tod_mean") -> np.ndarray
```
- `ffill_tod_mean` (default, option a): forward-fill zeros up to 6 steps, then fill the rest with `tod_mean[slot, sensor]`. If a `(slot, sensor)` has no train observations, fall back to the sensor's train mean, then the global train mean.
- `zero_mask` (option b, DCRNN convention): return speed unchanged.
- **Targets are never imputed.** Only model inputs pass through `impute_inputs`.
- Forward-fill may carry a value across a split boundary (e.g. last train value into the first val step). This is fine: it only uses the *past*, so it does not leak. Say so in the docstring.

---

## 1.5 Split — `src/traffic/data/split.py`

```python
@dataclass
class SplitIndex: train: slice; val: slice; test: slice

def chronological_split(T, ratios=(0.7, 0.1, 0.2)) -> SplitIndex
```
- Use `n_train = int(T * 0.7)`, `n_val = int(T * 0.1)`, `n_test = T - n_train - n_val`. Print the boundary timestamps.
- **Strict protocol (default):** windows are built inside each split separately (1.7).
- **DCRNN protocol (`split_protocol: dcrnn`):** build all windows over the whole series, then split *samples* 70/10/20 using DCRNN's rounding: `num_samples = T - 23 = 34249`, `num_test = round(0.2*n) = 6850`, `num_train = round(0.7*n) = 23974`, `num_val = n - train - test = 3425`. Assert these numbers — they match DCRNN's published `.npz` shapes. Label any result from this protocol as "literature-comparable".

---

## 1.6 Scaling — `src/traffic/data/scaling.py`

```python
class MaskedStandardScaler:
    def fit(self, x, mask, per_sensor=False) -> "MaskedStandardScaler"   # observed (mask==1) values only
    def transform(self, x) -> np.ndarray
    def inverse_transform(self, x) -> np.ndarray
```
- Fit **only** on `speed[train]` with `mask[train]`. Zeros excluded.
- Guard `std == 0` (constant sensor) → use 1.0 and log it.
- Missing input positions under `zero_mask`: scaling a raw 0 gives `(0 − mean)/std` ≈ −3, a fake "very slow" value. Rule: scale only observed/imputed values, set missing positions to 0.0 **in scaled space** (= the train mean), and let the `is_missing` channel carry the outage signal. Document in ADR-003.

---

## 1.7 Time features — `src/traffic/features/temporal.py` (needed by windows)

```python
def time_features(index: pd.DatetimeIndex) -> pd.DataFrame
# tod_sin, tod_cos (2π·minute_of_day/1440), dow_sin, dow_cos (2π·dow/7), is_weekend, is_holiday
```
`is_holiday` via `holidays.US(years=2012)`; within range this flags only Memorial Day 2012-05-28. Assert that in a test.

---

## 1.8 Windows — `src/traffic/data/windows.py`

```python
def make_windows(X_in, Y, Ymask, index, H=12, F=12, start=0, stop=None) -> Windows
# For each anchor t with start+H-1 <= t and t+F < stop:
#   X[t]  = X_in[t-H+1 : t+1]     -> [H, N, C]
#   Y[t]  = Y[t+1 : t+F+1]        -> [F, N]       (original mph)
#   M[t]  = Ymask[t+1 : t+F+1]    -> [F, N]
#   input_ts[t], target_ts[t]     -> timestamps for slicing + leakage tests
```
- Channels `C`: `[scaled_speed, is_missing, tod_sin, tod_cos, dow_sin, dow_cos, is_weekend, is_holiday]` — order fixed in one constant `CHANNELS`.
- Use `numpy.lib.stride_tricks.sliding_window_view` (no Python loop over T) and keep memory in mind: full METR-LA train X is ~24k × 12 × 207 × 8 float32 ≈ 1.9 GB. Offer a lazy mode (store anchors only; slice on the fly in the Dataset).
- Strict protocol: call once per split with `start/stop` = the split's slice. Windows never cross a boundary.

`src/traffic/data/datasets.py`: `TrafficWindowDataset(torch.utils.data.Dataset)` returning `(X, Y, M, target_idx)`; `make_loaders(cfg)` → train (shuffle *within* train is fine) / val / test (no shuffle). `--debug` uses `cfg.debug.n_sensors` and `n_days`.

Cache processed arrays to `data/processed/{protocol}_{imputation}_{H}_{F}.npz` with the config hash in the filename.

---

## 1.9 Tests (all must pass)

| Test file | What it checks |
|---|---|
| `test_load.py` | shapes, date range, 5-min regular index, ID alignment (real data, skip if absent); ID normalization on synthetic |
| `test_graph.py` | rebuild ≈ pkl (real data); direction convention on a hand-built 3-node chain; every variant has the expected shape/edge count; `random` keeps edge count; `identity` has no off-diagonal edges |
| `test_split_no_leakage.py` | strict: `max(train target_ts) < min(val input_ts)` and `max(val target_ts) < min(test input_ts)`; DCRNN protocol gives 23974/3425/6850 on real data |
| `test_scaling_train_only.py` | change val/test values (×10, add noise) → scaler params unchanged; zeros don't affect mean/std; tod_means unchanged when val/test change |
| `test_windows.py` | hand-built tiny series: X/Y alignment exact (Y[0] == series[H : H+F]); no window crosses split; channel order matches `CHANNELS` |

---

## 1.10 Script entry

`scripts/prepare_data.py` (add to tree; called by `make data` after download): loads, verifies, rebuilds graph, prints split boundaries + graph stats + missing stats, writes processed cache.

---

## Acceptance (M1)

- [ ] `make data` downloads (or prints manual instructions) and writes `MANIFEST.json`
- [ ] All data tests pass (synthetic in CI, real locally)
- [ ] Graph rebuild matches `adj_mx.pkl` (max abs diff < 1e-6) or the difference is documented in an ADR
- [ ] Split boundary timestamps printed and saved to `results/metrics/split_boundaries.json`
- [ ] ADR-002 (direction), ADR-003 (missing input scaling) written

**Commit:** `M1: data download, loaders, graph rebuild, leak-free split/scale/windows`
**STOP for review.**
