"""Out-of-fold XGBoost base forecasts for the residual hybrid (forward-chaining CV).

Train anchors are split into 5 chronological blocks B1..B5. For block k >= 2 an XGBoost
model trained on blocks < k (with a gap of H+F-1 anchors) predicts block k; B1 has no OOF
prediction and is dropped from residual training. The FINAL xgb_global models (trained on
all of train) predict val and test. Output: results/hybrid/xgb_base.npz

Usage: python scripts/make_oof.py [--debug]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml

from traffic.config import load_config
from traffic.data.pipeline import prepare
from traffic.features.build_tabular import build_tabular, subsample_rows, targets
from traffic.models import ml_models as mm
from traffic.models.hybrid import oof_folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    cfg = load_config("configs/ml/xgb_global.yaml")
    mm.N_JOBS = load_config("configs/base.yaml")["resources"]["n_jobs"]
    best = Path("configs/ml/xgb_global.best.yaml")
    if best.exists():
        cfg["model"]["params"].update(yaml.safe_load(best.read_text())["model"]["params"])
    p = prepare(cfg, cfg["paths"]["raw"], debug=args.debug)
    h_, f, n = cfg["history_steps"], cfg["horizon_steps"], p.x_in.shape[1]
    tabs = {s: build_tabular(p, s, cfg, "global", False) for s in ("train", "val", "test")}
    a_tr = len(tabs["train"].anchors)
    folds = oof_folds(a_tr, 5, h_ + f - 1)
    edges = np.array([fd[1].start for fd in folds] + [a_tr])
    params = dict(cfg["model"]["params"])
    frac = 0.15  # lighter than the final models: OOF only needs realistic, not optimal, errors
    oof = np.full((a_tr, f, n), np.nan, dtype=np.float32)
    flat = tabs["train"].flat()
    t0 = time.time()
    for h in range(1, f + 1):
        y, m = targets(p, "train", h)
        yv, mv = targets(p, "val", h)
        for tr_end, blk_anchors in folds:  # predict block from earlier blocks (+ gap)
            rows = np.arange(tr_end * n)
            idx = subsample_rows(m.ravel()[rows], None, frac, cfg["seed"])
            va = subsample_rows(mv.ravel(), 20000, 1.0, cfg["seed"])
            model = mm.fit_xgb(
                flat[idx],
                y.ravel()[idx],
                tabs["val"].flat()[va],
                yv.ravel()[va],
                params,
                tabs["train"].categorical,
                30,
                seed=cfg["seed"],
            )
            blk = slice(blk_anchors.start * n, blk_anchors.stop * n)
            oof[blk_anchors, h - 1] = model.predict(flat[blk]).reshape(-1, n)
        print(f"  OOF h={h:2d} done ({time.time() - t0:.0f}s)")
    # final models predict val and test
    val = np.zeros((len(tabs["val"].anchors), f, n), np.float32)
    test = np.zeros((len(tabs["test"].anchors), f, n), np.float32)
    ck = Path("checkpoints/ml" + ("_debug" if args.debug else ""))
    for h in range(1, f + 1):
        mdl = xgb.XGBRegressor()
        mdl.load_model(ck / f"xgb_global_h{h}.json")
        val[:, h - 1] = mdl.predict(tabs["val"].flat()).reshape(-1, n)
        test[:, h - 1] = mdl.predict(tabs["test"].flat()).reshape(-1, n)
    out = Path(cfg["paths"]["results"]) / ("debug" if args.debug else "") / "hybrid"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "xgb_base.npz", oof_train=oof, val=val, test=test, block_edges=edges)
    print("saved", out / "xgb_base.npz", "OOF coverage:", float((~np.isnan(oof[:, 0, 0])).mean()))


if __name__ == "__main__":
    main()
