"""Every model trains/predicts for a moment on the tiny --debug config (needs the real data)."""

import subprocess
import sys

import pytest

CMDS = [
    ["scripts/run_baselines.py", "--debug"],
    ["scripts/train_ml.py", "--model", "xgb_per_segment", "--debug"],
    ["scripts/train_ml.py", "--model", "xgb_global", "--debug"],
    ["scripts/train_ml.py", "--model", "rf", "--debug"],
    ["scripts/train_ml.py", "--model", "xgb_spatial", "--debug"],
    ["scripts/train_dl.py", "--model", "gru_nograph", "--debug"],
    ["scripts/train_dl.py", "--model", "stgat", "--debug"],
]


@pytest.mark.real_data
@pytest.mark.parametrize("cmd", CMDS, ids=[" ".join(c[:3]) for c in CMDS])
def test_debug_run(cmd):
    r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
