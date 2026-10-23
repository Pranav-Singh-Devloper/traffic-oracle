"""Train a deep model on several seeds and save test predictions.

Usage:
  python scripts/train_dl.py --model stgat --seeds 0 1 2
  python scripts/train_dl.py --model stgat --name stgat_a1_identity --out ablations \
        --set graph.variant=identity
  python scripts/train_dl.py --model stgat --debug
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from traffic.config import load_config
from traffic.data.pipeline import prepare
from traffic.evaluation.io import save_predictions
from traffic.logging_utils import log_run
from traffic.models.factory import build_model, channel_slice
from traffic.seed import get_device, set_seed
from traffic.training.trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gru_nograph", "stgat"])
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--name")
    ap.add_argument("--out", default="main", choices=["main", "ablations"])
    ap.add_argument("--set", nargs="*", default=[], help="key.sub=value overrides")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    cfg = load_config(f"configs/dl/{args.model}.yaml", args.set)
    name = args.name or args.model
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    if args.debug:
        cfg["train"]["max_epochs"] = 1
        seeds = seeds[:1]
    torch.set_num_threads(cfg["resources"]["torch_threads"])
    p = prepare(cfg, cfg["paths"]["raw"], debug=args.debug)
    device = get_device()
    sizes = {k: len(w) for k, w in p.windows.items()}
    print(f"{name}: device={device}, seeds={seeds}, windows={sizes}")
    root = Path(cfg["paths"]["results"]) / ("debug" if args.debug else "")
    pred_dir = root / ("ablations/predictions" if args.out == "ablations" else "predictions")
    ch = channel_slice(cfg)
    for seed in seeds:
        set_seed(seed)
        model = build_model(cfg, p.data.adj, p.data.locations, len(p.data.sensor_ids))
        n_par = sum(x.numel() for x in model.parameters())
        print(f"seed {seed}: {n_par:,} parameters")
        tr = Trainer(model, cfg["train"], p.scaler, device, ch, seed)
        t0 = time.time()
        hist = tr.fit(p.windows["train"], p.windows["val"])
        secs = time.time() - t0
        pred = tr.predict(p.windows["test"])
        w = p.windows["test"]
        y, m = w.targets()
        save_predictions(
            pred_dir / f"{name}_seed{seed}.npz",
            pred,
            y,
            m,
            w.target_ts(),
            p.data.sensor_ids,
            {
                "model": name,
                "seed": seed,
                "H": cfg["history_steps"],
                "n_params": n_par,
                "train_seconds": secs,
                "epochs": len(hist),
                "best_val_mae": float(hist.val_mae_mph.min()),
                "config": {k: cfg[k] for k in ("model", "train", "graph") if k in cfg},
            },
        )
        (root / "metrics" / "training").mkdir(parents=True, exist_ok=True)
        hist.to_csv(root / "metrics" / "training" / f"{name}_seed{seed}.csv", index=False)
        (root / "figures" / "training").mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist.epoch, hist.train_loss_scaled, label="train loss (scaled)")
        ax2 = ax.twinx()
        ax2.plot(hist.epoch, hist.val_mae_mph, color="C1", label="val MAE (mph)")
        ax.set(xlabel="epoch", title=f"{name} seed {seed}")
        fig.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "figures" / "training" / f"{name}_seed{seed}.png", dpi=120)
        plt.close(fig)
        if not args.debug:
            Path("checkpoints/dl").mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state": model.state_dict(), "cfg": cfg, "scaler": p.scaler.params()},
                f"checkpoints/dl/{name}_seed{seed}.pt",
            )
        log_run(
            f"{name}_seed{seed}",
            cfg,
            {
                "val_mae": round(float(hist.val_mae_mph.min()), 4),
                "epochs": len(hist),
                "seconds": round(secs),
            },
        )
        best_v = hist.val_mae_mph.min()
        print(f"seed {seed}: best val MAE {best_v:.3f}, {len(hist)} epochs, {secs:.0f}s")


if __name__ == "__main__":
    main()
