"""Who owns what (see OWNERS.yaml).

python scripts/owners.py check             every tracked file has an owner (or is shared)
python scripts/owners.py mine --as kunal   list one member's files
python scripts/owners.py commit --as kunal "message" [--dry-run]
      stage ONLY that member's changed files and commit with YOUR OWN git identity
      (user.name / user.email from your git config; nothing is faked)
python scripts/owners.py stats             commits per author on each member's files
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "OWNERS.yaml").read_text())


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


def owner_of(path: str) -> str | None:
    for key, m in CFG["members"].items():
        if any(fnmatch.fnmatch(path, p) for p in m["paths"]):
            return key
    if any(fnmatch.fnmatch(path, p) for p in CFG["shared"]):
        return "shared"
    return None


def tracked() -> list[str]:
    return [p for p in git("ls-files").splitlines() if p]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "mine", "commit", "stats"])
    ap.add_argument("message", nargs="?")
    ap.add_argument("--as", dest="who", choices=list(CFG["members"]))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="skip the git-email check")
    a = ap.parse_intermixed_args()
    files = tracked()
    if a.cmd == "check":
        orphans = [f for f in files if owner_of(f) is None]
        for f in orphans:
            print("NO OWNER:", f)
        counts = Counter(owner_of(f) for f in files)
        print(dict(counts))
        return 1 if orphans else 0
    if a.cmd == "mine":
        print("\n".join(f for f in files if owner_of(f) == a.who))
        return 0
    if a.cmd == "commit":
        if not (a.who and a.message):
            sys.exit('usage: owners.py commit --as <member> "message"')
        changed = [
            line[3:].strip().strip('"') for line in git("status", "--porcelain").splitlines()
        ]
        mine = [f for f in changed if owner_of(f) == a.who]
        other = [f for f in changed if owner_of(f) != a.who]
        who = f"{git('config', 'user.name').strip()} <{git('config', 'user.email').strip()}>"
        print("identity:", who)
        known = CFG["members"][a.who].get("emails", [])
        if known and git("config", "user.email").strip().lower() not in [e.lower() for e in known]:
            msg = (
                f"git user.email is not one of {known}; GitHub would not credit this commit "
                f"to {CFG['members'][a.who]['github']}. Fix: git config user.email <that email>"
            )
            if not (a.force or a.dry_run):
                sys.exit("STOP: " + msg)
            print("WARNING:", msg)
        print(f"will commit {len(mine)} file(s) owned by {a.who}; leave {len(other)} others alone")
        for f in mine:
            print("  +", f)
        if a.dry_run or not mine:
            return 0
        git("add", "--", *mine)
        git("commit", "-m", a.message, "--", *mine)
        print("committed. Now: git push origin <your-branch>")
        return 0
    stats: dict[str, Counter] = {k: Counter() for k in CFG["members"]}
    for f in files:
        o = owner_of(f)
        if o in stats:
            for line in git("log", "--format=%an", "--", f).splitlines():
                stats[o][line] += 1
    for k, c in stats.items():
        print(k, dict(c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
