PY ?= python
.PHONY: data eda phase1 phase2 report test lint debug all
data:   ; $(PY) scripts/download_data.py
eda:    ; $(PY) scripts/run_eda.py && $(PY) scripts/make_eda_summary.py
phase1: ; $(PY) scripts/run_baselines.py && $(PY) scripts/evaluate_all.py && $(PY) scripts/make_report_assets.py
phase2: ; @echo "not implemented yet (Phases 6-9)"
report: ; @echo "not implemented yet (Phases 5, 11)"
test:   ; $(PY) -m pytest -q
lint:   ; ruff check . && black --check .
debug:  ; $(PY) -m pytest -q tests/test_end_to_end_debug.py
all: data eda phase1 phase2 report
