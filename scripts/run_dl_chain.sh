#!/usr/bin/env bash
# Main DL runs, sequential and low priority (laptop-friendly). Usage: bash scripts/run_dl_chain.sh
set -e
echo "=== gru_nograph $(date)"; nice -n 10 python scripts/train_dl.py --model gru_nograph --seeds 0 1 2
echo "=== stgat $(date)";       nice -n 10 python scripts/train_dl.py --model stgat --seeds 0 1 2
echo "=== dl chain done $(date)"
