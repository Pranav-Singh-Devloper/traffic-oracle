"""Run the STGAT ablations A1-A12 (A13 = hybrid variants, read from Phase 7 outputs).

Each ablation changes ONE thing from the STGAT default through a config override. To keep the
laptop usable the budget is reduced: ``--epochs`` (default 25) and ``--seeds`` (default 0).
A reduced-budget reference run ``abl_default`` is trained with the same budget, so every
comparison is like-for-like. Predictions go to results/ablations/predictions/.

Usage:
  python scripts/run_ablations.py --list
  python scripts/run_ablations.py --ids A1 A2 --epochs 25 --seeds 0 [--resume] [--debug]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# id -> [(variant, [overrides])]
ABLATIONS: dict[str, list[tuple[str, list[str]]]] = {
    "A0": [("default", [])],
    "A1": [("identity", ["graph.variant=identity"])],
    "A2": [("random", ["graph.variant=random"])],
    "A3": [("forward", ["graph.variant=forward"]), ("backward", ["graph.variant=backward"])],
    "A4": [("gatv2", ["model.attention=gatv2"]), ("mean", ["model.attention=mean"])],
    "A5": [(f"heads{k}", [f"model.heads={k}"]) for k in (1, 2, 8)],
    "A6": [(f"layers{k}", [f"model.layers={k}"]) for k in (1, 3)],
    "A7": [("no_node_emb", ["model.node_embedding=false"])],
    "A8": [("no_time", ["model.use_time_features=false"])],
    "A9": [("zero_mask", ["input_imputation=zero_mask"])],
    "A10": [
        ("no_dropout", ["model.dropout=0", "model.attn_dropout=0"]),
        ("no_wd", ["train.weight_decay=0"]),
    ],
    "A11": [(f"H{k}", [f"history_steps={k}"]) for k in (6, 24)],
    "A12": [("edge_bias", ["model.edge_bias=true"])],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=list(ABLATIONS))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    runs = [(i, v, o) for i in args.ids for v, o in ABLATIONS[i]]
    if args.list:
        for i, v, o in runs:
            print(i, v, o)
        return
    out = Path("results/debug" if args.debug else "results") / "ablations" / "predictions"
    for aid, variant, overrides in runs:
        name = "abl_default" if aid == "A0" else f"abl_{aid}_{variant}"
        todo = [
            s for s in args.seeds if not (args.resume and (out / f"{name}_seed{s}.npz").exists())
        ]
        if not todo:
            print("skip", name)
            continue
        cmd = [
            sys.executable,
            "scripts/train_dl.py",
            "--model",
            "stgat",
            "--name",
            name,
            "--out",
            "ablations",
            "--seeds",
            *map(str, todo),
            "--set",
            f"train.max_epochs={args.epochs}",
            *overrides,
        ]
        if args.debug:
            cmd.append("--debug")
        print("RUN", name, overrides, flush=True)
        subprocess.run(["nice", "-n", "10", *cmd], check=True)


if __name__ == "__main__":
    main()
