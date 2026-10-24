"""One command per member: verify your own module on your own machine and commit the result.

  python scripts/first_commit.py --as kunal

It (1) checks your git identity matches OWNERS.yaml, (2) runs the tests for YOUR modules,
(3) appends the real outcome (date, identity, test result) to docs/viva/<you>.md, and
(4) commits only your files with YOUR git identity. Then push the printed branch.
Nothing is faked: if tests fail or the identity is wrong, it stops.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "OWNERS.yaml").read_text())["members"]
TESTS = {
    "pranav_singh": ["tests/test_gat_layer.py", "tests/test_stgat_shapes.py"],
    "kunal": [
        "tests/test_features_no_future.py",
        "tests/test_tabular_alignment.py",
        "tests/test_hybrid.py",
    ],
    "himanshu": ["tests/test_metrics_masking.py", "tests/test_baselines.py", "tests/test_ctm.py"],
    "kanchan": [
        "tests/test_load.py",
        "tests/test_graph.py",
        "tests/test_split_no_leakage.py",
        "tests/test_scaling_train_only.py",
        "tests/test_windows.py",
    ],
    "vaibhav": ["tests"],
}


def run(*cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as", dest="who", required=True, choices=list(CFG))
    a = ap.parse_args()
    m = CFG[a.who]
    email = run("git", "config", "user.email").stdout.strip()
    name = run("git", "config", "user.name").stdout.strip()
    if email.lower() not in [e.lower() for e in m["emails"]]:
        sys.exit(
            f"STOP: git user.email is '{email}', expected one of {m['emails']}.\n"
            f"Fix: git config user.email {m['emails'][0]}"
        )
    print(f"identity OK: {name} <{email}> for GitHub {m['github']}")
    t = run(sys.executable, "-m", "pytest", "-q", *TESTS[a.who])
    last = t.stdout.strip().splitlines()[-1] if t.stdout.strip() else "no output"
    print("tests:", last)
    if t.returncode != 0:
        sys.exit("STOP: tests failed; fix or tell Pranav before committing.\n" + t.stdout[-1500:])
    branch = f"{a.who}/first-verification"
    run("git", "switch", "-c", branch)
    f = ROOT / "docs" / "viva" / f"{a.who}.md"
    f.write_text(
        f.read_text().rstrip("\n")
        + f"\n\n## Verification log\n- {datetime.now():%Y-%m-%d %H:%M}: {m['name']} ran the "
        f"tests for their modules ({', '.join(TESTS[a.who])}): {last}.\n"
    )
    c = run(
        sys.executable,
        "scripts/owners.py",
        "commit",
        "--as",
        a.who,
        f"{m['name']}: verify own modules on own machine ({last})",
    )
    print(c.stdout, c.stderr)
    if c.returncode != 0:
        return 1
    print(f"Done. Now run:  git push -u origin {branch}   and open a pull request.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
