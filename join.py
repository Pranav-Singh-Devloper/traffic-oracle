#!/usr/bin/env python3
"""Team T042: ONE command after cloning.   python3 join.py

It does everything for you, on your own machine and under your own GitHub identity:
  1. figures out who you are (GitHub login / git email) and sets your git name+email
  2. creates the Python environment and installs the project
  3. downloads the dataset
  4. runs the checks for YOUR modules (nothing is faked: failures stop the script)
  5. records the real result in docs/viva/<you>.md and commits it as you
  6. pushes your branch and opens the pull request

Only needs: git and Python 3 installed. Safe to run again (it stops if you already joined).
Options: --dry-run (do everything except commit/push), --me <key> (skip auto-detection).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = "Pranav-Singh-Devloper/traffic-oracle"
TEAM = {
    "pranav_singh": (
        "Pranav Singh",
        "Pranav-Singh-Devloper",
        ["raindeer.social@gmail.com", "pranavsinghprimo@gmail.com"],
    ),
    "kunal": ("Kunal", "kunaldevsahu", ["kunaldevsahu5@gmail.com"]),
    "himanshu": ("Himanshu", "HimanshuPal29", ["himanshu.pal2024@nst.rishihood.edu.in"]),
    "kanchan": ("Kanchan", "kanchan97-cp", ["bansalkanchan23@gmail.com"]),
    "vaibhav": ("Vaibhav", "VaibhavSingh50", ["vaibhavsingh0564@gmail.com"]),
}
# role -> (tests, extra debug command run with the project python). All need only the dataset.
CHECKS = {
    "pranav_singh": (
        ["tests/test_gat_layer.py", "tests/test_stgat_shapes.py"],
        ["scripts/train_dl.py", "--model", "stgat", "--debug"],
    ),
    "kunal": (
        [
            "tests/test_features_no_future.py",
            "tests/test_tabular_alignment.py",
            "tests/test_hybrid.py",
        ],
        ["scripts/train_ml.py", "--model", "xgb_global", "--debug"],
    ),
    "himanshu": (
        ["tests/test_metrics_masking.py", "tests/test_baselines.py", "tests/test_ctm.py"],
        ["scripts/run_baselines.py", "--debug"],
    ),
    "kanchan": (
        [
            "tests/test_load.py",
            "tests/test_graph.py",
            "tests/test_split_no_leakage.py",
            "tests/test_scaling_train_only.py",
            "tests/test_windows.py",
        ],
        ["scripts/prepare_data.py", "--debug"],
    ),
    "vaibhav": (["tests"], None),
}
WIN = os.name == "nt"
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if WIN else "bin/python")


def sh(*cmd: str, check: bool = True, quiet: bool = False, timeout: int | None = None):
    r = subprocess.run(list(cmd), cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        sys.exit(f"\nSTOP: command failed: {' '.join(cmd)}\n{(r.stdout + r.stderr)[-1800:]}")
    if not quiet and r.stdout.strip():
        print(r.stdout.strip()[-600:])
    return r


def say(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def detect_member(forced: str | None) -> str:
    if forced:
        return forced
    gh = shutil.which("gh")
    if gh:
        r = subprocess.run(
            [gh, "api", "user", "--jq", ".login"], capture_output=True, text=True, timeout=20
        )
        login = r.stdout.strip().lower()
        for k, (_, g, _) in TEAM.items():
            if g.lower() == login:
                print(f"Detected from GitHub CLI login: {login}")
                return k
    email = sh("git", "config", "user.email", check=False, quiet=True).stdout.strip().lower()
    for k, (_, _, emails) in TEAM.items():
        if email in [e.lower() for e in emails]:
            print(f"Detected from git email: {email}")
            return k
    print("Who are you?")
    keys = list(TEAM)
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {TEAM[k][0]}  (GitHub: {TEAM[k][1]})")
    while True:
        c = input("Type your number: ").strip()
        if c.isdigit() and 1 <= int(c) <= len(keys):
            k = keys[int(c) - 1]
            if (
                input(f"Confirm you are {TEAM[k][0]} and own {TEAM[k][2][0]}? [y/N] ").lower()
                == "y"
            ):
                return k


def setup_env() -> str:
    def has_pkg() -> bool:
        return (
            VENV_PY.exists()
            and subprocess.run(
                [str(VENV_PY), "-c", "import traffic, pytest"], capture_output=True
            ).returncode
            == 0
        )

    if has_pkg():
        return str(VENV_PY)
    say("Creating the environment (first time only, a few minutes)")
    uv = shutil.which("uv")
    if uv:
        uv_cmd = [uv]
    else:  # tiny throw-away venv holding only 'uv' (works even on locked-down system Pythons)
        boot = ROOT / ".uv-bootstrap"
        print("Installing the 'uv' helper ...")
        sh(sys.executable, "-m", "venv", str(boot), "--clear")
        bpy = boot / ("Scripts/python.exe" if WIN else "bin/python")
        sh(str(bpy), "-m", "pip", "install", "-q", "uv")
        uv_cmd = [str(bpy), "-m", "uv"]
    sh(*uv_cmd, "venv", "--python", "3.11", ".venv", "--clear")
    sh(*uv_cmd, "pip", "install", "--python", str(VENV_PY), "-e", ".[dev]", timeout=3600)
    if not has_pkg():
        sys.exit("STOP: environment install did not work; send the output above to Pranav.")
    return str(VENV_PY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", choices=list(TEAM))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if sh("git", "rev-parse", "--is-inside-work-tree", check=False, quiet=True).returncode != 0:
        sys.exit("Run this inside the cloned traffic-oracle folder.")
    key = detect_member(a.me)
    name, github, emails = TEAM[key]
    say(f"Hello {name} ({github})")

    cur = sh("git", "config", "user.email", check=False, quiet=True).stdout.strip().lower()
    if cur not in [e.lower() for e in emails]:
        sh("git", "config", "user.email", emails[0])
        print(f"Set git email for this repo to {emails[0]} (the one registered on your GitHub).")
    sh("git", "config", "user.name", name)

    branch = f"{key}/first-verification"
    sh("git", "fetch", "origin", quiet=True, check=False)
    if sh("git", "ls-remote", "--heads", "origin", branch, check=False, quiet=True).stdout.strip():
        print(f"\nYou have already joined: branch {branch} exists on GitHub. Nothing more to do.")
        return 0
    sh("git", "switch", "main", check=False, quiet=True)
    sh("git", "pull", "--ff-only", "origin", "main", check=False, quiet=True)

    py = setup_env()
    if not (ROOT / "data/raw/metr-la.h5").exists():
        say("Downloading the dataset (about 60 MB)")
        sh(py, "scripts/download_data.py", timeout=1800)

    tests, extra = CHECKS[key]
    say("Running checks for your modules")
    t = sh(py, "-m", "pytest", "-q", *tests, check=False, quiet=True)
    last = (t.stdout.strip().splitlines() or ["no output"])[-1]
    print("tests:", last)
    if t.returncode != 0:
        sys.exit("STOP: tests failed:\n" + t.stdout[-1500:])
    ran = ""
    if extra:
        e = sh(py, *extra, check=False, quiet=True)
        ran = f" and `python {' '.join(extra)}`"
        print(f"{' '.join(extra)}:", "OK" if e.returncode == 0 else "FAILED")
        if e.returncode != 0:
            sys.exit("STOP: your debug run failed:\n" + (e.stdout + e.stderr)[-1500:])
    lint = sh(py, "-m", "ruff", "check", ".", check=False, quiet=True)
    print("lint:", "clean" if lint.returncode == 0 else "has warnings (ok for now)")

    viva = ROOT / "docs" / "viva" / f"{key}.md"
    viva.write_text(
        viva.read_text().rstrip("\n")
        + f"\n\n## Verification log\n- {datetime.now():%Y-%m-%d %H:%M}: {name} ran "
        f"`{' '.join(tests)}` ({last}){ran} on their own machine.\n"
    )
    if a.dry_run:
        sh("git", "checkout", "--", str(viva.relative_to(ROOT)), check=False, quiet=True)
        print(
            "\nDRY RUN finished: everything works. Run again without --dry-run to commit and push."
        )
        return 0
    sh("git", "switch", "-c", branch)
    sh("git", "add", "--", str(viva.relative_to(ROOT)))
    sh("git", "commit", "-m", f"{name}: verify own modules on own machine ({last})")
    say("Pushing your branch")
    p = sh("git", "push", "-u", "origin", branch, check=False, quiet=True)
    if p.returncode != 0:
        sys.exit(
            "STOP: push failed (are you added as a collaborator and logged in to GitHub?):\n"
            + p.stderr[-1200:]
        )
    url = f"https://github.com/{REPO}/compare/main...{branch}?expand=1"
    if shutil.which("gh"):
        r = sh(
            "gh",
            "pr",
            "create",
            "--repo",
            REPO,
            "--base",
            "main",
            "--head",
            branch,
            "--title",
            f"{name}: verify own modules",
            "--body",
            f"Ran `python3 join.py` on my machine: {last}.",
            check=False,
            quiet=True,
        )
        url = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else url
    print(f"\nALL DONE, {name}!  Your commit is on GitHub under your own account.")
    print(f"Pull request: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
