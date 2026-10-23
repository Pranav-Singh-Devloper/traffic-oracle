"""Append run metadata (config hash, git hash, command) to docs/experiment_log.md."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LOG = Path("docs/experiment_log.md")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def log_run(name: str, config: dict, metrics: dict | None = None) -> None:
    """Append one row: time, name, git hash (+dirty), config hash, command, metrics."""
    commit = _git("rev-parse", "--short", "HEAD") or "uncommitted"
    dirty = "+dirty" if _git("status", "--porcelain") else ""
    cfg_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    row = (
        f"| {datetime.now():%Y-%m-%d %H:%M} | {name} | {commit}{dirty} | {cfg_hash} | "
        f"`{' '.join(sys.argv)}` | {json.dumps(metrics or {})} |\n"
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(row)
