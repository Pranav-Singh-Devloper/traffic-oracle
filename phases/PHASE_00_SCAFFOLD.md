# Phase 0 — Scaffold (M0)

**Goal:** an empty but working project: structure, configs, seeding, lint, tests, CI. No data or models yet.
**SPEC refs:** §4 (structure + stack), §5.1 (base config), §19 (code quality), §14 M0.
**Prerequisites:** none.
**Owner:** Vaibhav (scaffold, CI, README); reviewed by Pranav Singh

---

## 0.1 Repository decision (ADR-001)

`DL_AML_Project/` is nested in the parent git repo `folder_2/`. Before writing code:
- Recommended: `git init` inside `DL_AML_Project/` and push to a new GitHub repo `traffic-gnn-forecasting`, so CI and per-member history work.
- **Ask the human before running `git init`** (it changes how the parent repo sees this folder).
- Record the decision in `docs/decisions.md` as ADR-001.
- Rename or symlink `SPEC.md` → `PROJECT_SPEC.md` (SPEC §4 uses that name). Keep one copy only.

## 0.2 Create the directory tree

Create every folder from SPEC §4 with `__init__.py` in each Python package:

```
src/traffic/{data,features,models,models/layers,training,evaluation,viz,control,control/sumo}
configs/{ml,dl,hybrid,ablations}
scripts/  tests/  notebooks/  slides/
results/{predictions,metrics,figures,maps}
report/{phase1,phase2,figures,tables}
docs/{theory,viva}
data/{raw,processed}
```

Put a one-line docstring in each module stub saying what it will hold (e.g. `"""Masked StandardScaler fit on training timestamps only."""`).

## 0.3 Packaging and dependencies

`pyproject.toml` (setuptools, `src/` layout, package `traffic`, Python `>=3.11,<3.12`) plus a pinned `requirements.txt`:

```
numpy pandas scipy scikit-learn xgboost torch torch-geometric networkx
folium matplotlib optuna pyyaml tables holidays shap statsmodels joblib
gdown requests tqdm pytest ruff black pre-commit
```
- Pin exact versions that install together on Python 3.11 (resolve once, then freeze).
- `tables` is required by `pd.read_hdf`.
- `gdown` is for the Google Drive download of `metr-la.h5`.
- `torch-geometric` is used only for the GAT equivalence test and optional DCRNN; our own GAT layer must not need it.
- Optional extras: `contextily` (static basemaps), `sumo`/`traci` (control alt), `torch-geometric-temporal`.

Install editable: `pip install -e .`

## 0.4 Configs

`configs/base.yaml` — copy SPEC §5.1 exactly, plus:
```yaml
paths:
  raw: data/raw
  processed: data/processed
  results: results
input_imputation: ffill_tod_mean   # (a) default | zero_mask (b)
max_ffill_steps: 6
scaler: global                     # global | per_sensor
split_protocol: strict             # strict | dcrnn
use_daily_lag: false
debug:
  n_sensors: 10
  n_days: 2
  epochs: 1
```
Create empty-but-valid stubs for every model config in SPEC §4 (`ml/*.yaml`, `dl/*.yaml`, `hybrid/residual_hybrid.yaml`). Each model config has `inherits: base.yaml` and a `model:` block.

`src/traffic/config.py`:
- `load_config(path, overrides: list[str] | None) -> Config` — loads YAML, merges `inherits`, applies `key.sub=value` CLI overrides.
- Validate with dataclasses (`split` sums to 1.0, `horizon_steps >= max(report_horizons)`, etc.). Fail loudly on unknown keys.

## 0.5 Seeding and device

`src/traffic/seed.py`:
- `set_seed(seed)` seeds `random`, `numpy`, `torch`, `torch.cuda`; sets `PYTHONHASHSEED`.
- `set_deterministic(flag)` sets `torch.backends.cudnn.deterministic`, `benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`.
- `get_device()` returns `cuda` → `mps` → `cpu`.

## 0.6 Run logging

`src/traffic/logging_utils.py` (small helper, add to the tree):
- `log_run(name, config, metrics: dict | None)` appends a row to `docs/experiment_log.md` with timestamp, git commit hash (`git rev-parse --short HEAD`, or `"uncommitted"`), dirty flag, config hash, and the command line.

## 0.7 Makefile

```make
data:     python scripts/download_data.py
eda:      python scripts/run_eda.py
phase1:   (baselines + all ML models + evaluate_all)
phase2:   (all DL + hybrid + ablations + maps + evaluate_all)
report:   python scripts/make_report_assets.py && latexmk -pdf ...
test:     pytest -q
lint:     ruff check . && black --check .
debug:    pytest -q tests/test_end_to_end_debug.py
all:      data eda phase1 phase2 report
```
Targets that need later phases print `"not implemented yet (Phase N)"` and exit 0 for now.

## 0.8 Lint, format, pre-commit

- `ruff` (rules: E, F, I, B, UP, N), `black` line length 100, configured in `pyproject.toml`.
- `.pre-commit-config.yaml` with ruff + black + end-of-file + trailing whitespace.

## 0.9 CI

`.github/workflows/ci.yml`: Ubuntu, Python 3.11, CPU torch wheel, cache pip, run `make lint` and `make test`.
CI never downloads the real dataset. Tests that need data use a **synthetic fixture** (see 0.10) or are skipped with `pytest.mark.skipif(not data_present)`.

## 0.10 Test fixtures

`tests/conftest.py`:
- `synthetic_metr` fixture: a small DataFrame shaped like METR-LA (DatetimeIndex at 5-min, 10 sensor columns as strings, ~3 days, sinusoidal daily speed 20–65 mph, ~5% zeros injected, one 2-hour all-sensor outage).
- `synthetic_graph` fixture: 10×10 asymmetric weighted adjacency + sensor id list.
- `real_data` marker that skips when `data/raw/metr-la.h5` is missing.

Create stub test files from SPEC §4 with one placeholder test each (`assert True`) so `make test` runs.

## 0.11 Docs templates

- `README.md`: title, one-paragraph summary, "how to run" (`make all`), hardware/runtime placeholders, and the markers `<!-- RESULTS_START -->` / `<!-- RESULTS_END -->`.
- `CONTRIBUTIONS.md`: table `Member | Modules owned | Experiments run | Report sections` with SPEC §16 suggested split; leave names as `TODO`.
- `docs/decisions.md`: ADR template (`Context / Decision / Consequences / Status`) + ADR-001 (repo root).
- `docs/experiment_log.md`: header row.
- `data/README.md`: provenance (DCRNN repo + Google Drive link), license note, file list.
- `.gitignore`: `data/*` except `data/README.md` and `data/raw/MANIFEST.json`; `results/predictions/`; checkpoints (`*.pt`); `.venv`; LaTeX build files.
- `LICENSE` (MIT unless team says otherwise).

---

## Acceptance (M0)

- [ ] `pip install -e .` works on a fresh Python 3.11 venv
- [ ] `make test` passes (placeholder tests)
- [ ] `make lint` passes
- [ ] `python -c "from traffic.config import load_config; print(load_config('configs/base.yaml'))"` prints the config
- [ ] CI workflow green on GitHub (or, if no remote yet, `act`/local run documented)
- [ ] ADR-001 written

**Commit:** `M0: project scaffold, configs, CI`
**STOP for review.**
