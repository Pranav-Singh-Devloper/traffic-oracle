#!/usr/bin/env bash
# Reduced-budget STGAT ablations (1 seed, 12 epochs), laptop-friendly. Resumable.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -u scripts/run_ablations.py --epochs 12 --seeds 0 --resume
echo "=== ablation chain done $(date)"
