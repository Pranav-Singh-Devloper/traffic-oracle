"""Download METR-LA + DCRNN sensor-graph files into data/raw and write MANIFEST.json.

Usage: python scripts/download_data.py [--verify-only]
Source: https://github.com/liyaguang/DCRNN#data-preparation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

RAW = Path("data/raw")
GH = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/"
GITHUB_FILES = [
    "adj_mx.pkl",
    "distances_la_2012.csv",
    "graph_sensor_ids.txt",
    "graph_sensor_locations.csv",
]
DRIVE_FOLDER_ID = "10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
REQUIRED = ["metr-la.h5", *GITHUB_FILES]

MANUAL = """Could not download METR-LA automatically.
1. Open https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX
   (or Baidu: https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) and download metr-la.h5
2. Download the sensor graph files from
   https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph
3. Place them at:
   data/raw/metr-la.h5
   data/raw/adj_mx.pkl
   data/raw/distances_la_2012.csv
   data/raw/graph_sensor_ids.txt
   data/raw/graph_sensor_locations.csv
4. Re-run: python scripts/download_data.py --verify-only
"""


def fetch(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> bool:
    """Stream ``url`` to ``dest`` with retries and backoff."""
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2**attempt)
    return False


def fetch_h5(dest: Path) -> bool:
    """Try gdown on the Drive folder; pick metr-la.h5 out of it."""
    try:
        import gdown

        with tempfile.TemporaryDirectory() as tmp:
            gdown.download_folder(id=DRIVE_FOLDER_ID, output=tmp, quiet=True)
            hits = list(Path(tmp).rglob("metr-la.h5"))
            if not hits:
                return False
            shutil.move(str(hits[0]), dest)
            return True
    except Exception as e:  # noqa: BLE001
        print(f"  gdown failed: {e}")
        return False


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify() -> bool:
    """Shape checks from SPEC section 2.4 (needs pandas + tables)."""
    from traffic.data.load import load_all

    try:
        load_all(RAW)
    except Exception as e:  # noqa: BLE001
        print(f"Integrity check FAILED: {e}")
        return False
    print("Integrity checks passed.")
    return True


def write_manifest(sources: dict[str, str]) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    m = {
        n: {
            "bytes": (RAW / n).stat().st_size,
            "sha256": sha256(RAW / n),
            "source_url": sources.get(n, "manual"),
            "downloaded_at": now,
        }
        for n in REQUIRED
    }
    (RAW / "MANIFEST.json").write_text(json.dumps(m, indent=2))
    print("Wrote", RAW / "MANIFEST.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    sources: dict[str, str] = {}
    if not args.verify_only:
        for name in GITHUB_FILES:
            if (RAW / name).exists():
                continue
            print("Downloading", name)
            fetch(GH + name, RAW / name)
            sources[name] = GH + name
        if not (RAW / "metr-la.h5").exists():
            print("Downloading metr-la.h5 from Google Drive (gdown)")
            if fetch_h5(RAW / "metr-la.h5"):
                sources["metr-la.h5"] = f"https://drive.google.com/open?id={DRIVE_FOLDER_ID}"
    missing = [n for n in REQUIRED if not (RAW / n).exists()]
    if missing:
        print("Missing:", missing)
        print(MANUAL)
        return 1
    if not verify():
        return 1
    write_manifest(sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
