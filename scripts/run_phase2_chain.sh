#!/usr/bin/env bash
# After the DL chain: OOF base forecasts, hybrids, evaluation, maps, control. Low priority.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
until grep -q "=== dl chain done" logs/dl_chain.log; do sleep 30; done
run() { echo "=== $* $(date)"; nice -n 10 "$@"; }
run python scripts/evaluate_all.py
run python scripts/make_oof.py
run python -u scripts/train_hybrid.py --kind residual --seeds 0 1 2
run python scripts/train_hybrid.py --kind average --seeds 0 1 2
run python scripts/train_hybrid.py --kind gating --seeds 0 1 2
run python scripts/evaluate_all.py
run python scripts/make_maps.py
run python scripts/run_control.py
run python scripts/make_report_assets.py
echo "=== phase2 chain done $(date)"
