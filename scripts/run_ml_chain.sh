#!/usr/bin/env bash
# Tune then train all ML models sequentially (full data). Usage: bash scripts/run_ml_chain.sh
set -e
for m in xgb_global xgb_spatial xgb_per_segment rf; do
  echo "=== tune $m $(date)"; python scripts/train_ml.py --model $m --tune
  echo "=== train $m $(date)"; python scripts/train_ml.py --model $m
done
echo "=== chain done $(date)"
