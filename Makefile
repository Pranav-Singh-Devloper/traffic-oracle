PY ?= python
.PHONY: data eda phase1 phase2 report test lint debug all
data:   ; $(PY) scripts/download_data.py && $(PY) scripts/prepare_data.py
eda:    ; $(PY) scripts/run_eda.py && $(PY) scripts/make_eda_summary.py
phase1: ; $(PY) scripts/run_baselines.py && $(PY) scripts/evaluate_all.py && $(PY) scripts/make_report_assets.py
phase2: ; bash scripts/run_ml_chain.sh && bash scripts/run_dl_chain.sh && bash scripts/run_phase2_chain.sh
report: ; $(PY) scripts/make_report_assets.py && cd report/phase1 && tectonic -X compile main.tex && cd ../phase2 && tectonic -X compile main.tex
test:   ; $(PY) -m pytest -q
lint:   ; ruff check . && black --check .
debug:  ; $(PY) -m pytest -q tests/test_end_to_end_debug.py
all: data eda phase1 phase2 report
