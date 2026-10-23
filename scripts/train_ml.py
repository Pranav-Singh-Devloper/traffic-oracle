"""Train an ML model and save test predictions in the common format.

Usage:
  python scripts/train_ml.py --model xgb_global [--tune] [--debug]
  models: xgb_per_segment | xgb_global | rf | xgb_spatial
Tuned hyperparameters (if present) are read from configs/ml/<name>.best.yaml.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from traffic.config import load_config
from traffic.data.pipeline import prepare
from traffic.evaluation.io import save_predictions
from traffic.features.build_tabular import build_tabular, subsample_rows, targets
from traffic.logging_utils import log_run
from traffic.models import ml_models as mm
from traffic.training import tuning

NAMES = {
    "xgb_per_segment": "xgb_per_segment",
    "xgb_global": "xgb_global",
    "rf": "rf_multioutput",
    "xgb_spatial": "xgb_spatial",
}


def run_tuning(p, cfg, tabs, name: str, spatial: bool) -> None:
    mc = cfg["model"]
    tc = mc["tuning"]
    met = Path(cfg["paths"]["results"]) / "metrics"
    t0 = time.time()
    if name == "rf_multioutput":
        f = cfg["horizon_steps"]

        def stack(split):
            ys, ms = zip(*[targets(p, split, h) for h in range(1, f + 1)], strict=True)
            return np.stack(ys, -1).reshape(-1, f), np.stack(ms, -1).reshape(-1, f).all(1)

        ytr, mtr = stack("train")
        yva, mva = stack("val")
        itr = subsample_rows(mtr, tc["train_rows"], 1.0, cfg["seed"])
        iva = subsample_rows(mva, tc["val_rows"], 1.0, cfg["seed"])
        best, df = tuning.tune_rf(
            tabs["train"].flat()[itr],
            ytr[itr],
            tabs["val"].flat()[iva],
            yva[iva],
            tc["n_trials"],
            cfg["seed"],
        )
        extra = {"row_subsample": mc["row_subsample"]}
    else:
        h = tc["tune_horizon"]
        if name == "xgb_per_segment":  # pooled sample of 20 sensors, no static features
            sensors = np.random.default_rng(cfg["seed"]).choice(p.x_in.shape[1], 20, replace=False)
            ytr, mtr = targets(p, "train", h)
            yva, mva = targets(p, "val", h)
            xtr = tabs["train"].X[:, sensors].reshape(-1, tabs["train"].X.shape[-1])
            xva = tabs["val"].X[:, sensors].reshape(-1, tabs["val"].X.shape[-1])
            mt, mv = mtr[:, sensors].ravel(), mva[:, sensors].ravel()
            ytr, yva = ytr[:, sensors].ravel(), yva[:, sensors].ravel()
        else:
            ytr, mtr = targets(p, "train", h)
            yva, mva = targets(p, "val", h)
            xtr, xva = tabs["train"].flat(), tabs["val"].flat()
            mt, mv = mtr.ravel(), mva.ravel()
            ytr, yva = ytr.ravel(), yva.ravel()
        itr = subsample_rows(mt, tc["train_rows"], 1.0, cfg["seed"])
        iva = subsample_rows(mv, tc["val_rows"], 1.0, cfg["seed"])
        base = {"n_estimators": 800}
        best, df = tuning.tune_xgb(
            xtr[itr],
            ytr[itr],
            xva[iva],
            yva[iva],
            tabs["train"].categorical,
            tc["n_trials"],
            cfg["seed"],
            base,
            mc["early_stopping_rounds"],
        )
        extra = {}
    df.to_csv(met / f"tuning_{name}.csv", index=False)
    tuning.save_best(Path("configs/ml") / f"{name}.best.yaml", name, best, extra)
    print(f"tuned {name}: best val MAE {df['value'].min():.4f} in {time.time() - t0:.0f}s")
    print("params:", best)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(NAMES))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--no-tuned", action="store_true", help="ignore configs/ml/<name>.best.yaml")
    args = ap.parse_args()
    name = NAMES[args.model]
    cfg = load_config(f"configs/ml/{name}.yaml")
    best = Path(f"configs/ml/{name}.best.yaml")
    if best.exists() and not args.no_tuned and not args.tune:
        cfg["model"]["params"].update(yaml.safe_load(best.read_text())["model"]["params"])
        print("using tuned params from", best)
    if args.debug:
        cfg["model"]["params"] = {**cfg["model"]["params"], "n_estimators": 20}
        if "n_estimators" in cfg["model"]["params"] and name == "rf_multioutput":
            cfg["model"]["params"]["n_estimators"] = 10
        cfg["model"].get("tuning", {}).update({"n_trials": 2})
    p = prepare(cfg, cfg["paths"]["raw"], debug=args.debug)
    spatial = name == "xgb_spatial"
    kind = "segment" if name == "xgb_per_segment" else "global"
    t0 = time.time()
    tabs = {s: build_tabular(p, s, cfg, kind, spatial) for s in ("train", "val", "test")}
    print(
        f"features: {len(tabs['train'].names)} cols, train {tabs['train'].X.shape} "
        f"({time.time() - t0:.0f}s)"
    )
    (Path(cfg["paths"]["results"]) / "metrics").mkdir(parents=True, exist_ok=True)
    if args.tune:
        run_tuning(p, cfg, tabs, name, spatial)
        return

    ckpt = Path("checkpoints/ml" + ("_debug" if args.debug else ""))
    info: dict = {}
    t0 = time.time()
    if name == "xgb_per_segment":
        pred, _ = mm.train_xgb_per_segment(p, cfg, tabs)
    elif name == "rf_multioutput":
        pred, _, info = mm.train_rf(p, cfg, tabs)
    else:
        pred, _ = mm.train_xgb_pooled(p, cfg, tabs, ckpt)
    secs = time.time() - t0
    w = p.windows["test"]
    y, m = w.targets()
    res = Path(cfg["paths"]["results"]) / ("debug" if args.debug else "")
    out = res / "predictions" / f"{name}.npz"
    save_predictions(
        out,
        pred,
        y,
        m,
        w.target_ts(),
        p.data.sensor_ids,
        {
            "model": name,
            "H": cfg["history_steps"],
            "params": cfg["model"].get("params"),
            "train_seconds": secs,
            "features": tabs["train"].names,
            **info,
        },
    )
    log_run(name, cfg, {"train_seconds": round(secs)})
    print(f"saved {out} ({secs:.0f}s)")


if __name__ == "__main__":
    main()
