"""XGBoost (per-segment, global, spatial) and multi-output Random Forest.

All trainers return predictions shaped [S, F, N] (NaN for horizons not trained), the same
layout every other model uses, plus the fitted models for SHAP / analysis.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from traffic.data.pipeline import Prepared
from traffic.features.build_tabular import Tabular, subsample_rows, targets

N_JOBS = 6  # leave cores free for the rest of the machine (set from configs/base.yaml)


def make_xgb(
    params: dict,
    categorical: list[int],
    n_features: int,
    early_stopping: int | None,
    n_jobs: int,
    seed: int,
) -> xgb.XGBRegressor:
    types = ["c" if i in categorical else "q" for i in range(n_features)]
    return xgb.XGBRegressor(
        tree_method="hist",
        eval_metric="mae",
        n_jobs=n_jobs,
        random_state=seed,
        early_stopping_rounds=early_stopping,
        enable_categorical=bool(categorical),
        feature_types=types,
        max_cat_to_onehot=1,
        **params,
    )


def fit_xgb(
    xtr, ytr, xva, yva, params, categorical, early_stopping, n_jobs=None, seed=42
) -> xgb.XGBRegressor:
    m = make_xgb(params, categorical, xtr.shape[1], early_stopping, n_jobs or N_JOBS, seed)
    m.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
    return m


def _horizons(spec, f: int) -> list[int]:
    return list(range(1, f + 1)) if spec == "all" else list(spec)


def _flat_rows(p: Prepared, tab: Tabular, split: str, h: int, frac: float, n_rows, seed: int):
    y, m = targets(p, split, h)
    idx = subsample_rows(m.ravel(), n_rows, frac, seed)
    return tab.flat()[idx], y.ravel()[idx]


def train_xgb_pooled(
    p: Prepared,
    cfg: dict,
    tabs: dict[str, Tabular],
    out_dir: Path | None = None,
    n_jobs: int | None = None,
    log=print,
) -> tuple[np.ndarray, dict[int, xgb.XGBRegressor]]:
    """One pooled model per horizon (xgb_global / xgb_spatial)."""
    mc = cfg["model"]
    f, n = cfg["horizon_steps"], p.x_in.shape[1]
    s = len(tabs["test"].anchors)
    pred = np.full((s, f, n), np.nan, dtype=np.float32)
    models: dict[int, xgb.XGBRegressor] = {}
    for h in _horizons(mc["horizons"], f):
        t0 = time.time()
        xtr, ytr = _flat_rows(p, tabs["train"], "train", h, mc["row_subsample"], None, cfg["seed"])
        xva, yva = _flat_rows(p, tabs["val"], "val", h, 1.0, None, cfg["seed"])
        m = fit_xgb(
            xtr,
            ytr,
            xva,
            yva,
            mc["params"],
            tabs["train"].categorical,
            mc["early_stopping_rounds"],
            n_jobs,
            cfg["seed"],
        )
        pred[:, h - 1] = m.predict(tabs["test"].flat()).reshape(s, n)
        models[h] = m
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            m.save_model(out_dir / f"{mc['name']}_h{h}.json")
        log(f"  h={h:2d} rows={len(ytr):,} best_iter={m.best_iteration} {time.time() - t0:.0f}s")
    return pred, models


def _fit_segment(j, h, xtr, ytr, xva, yva, xte, params, es, seed):
    m = fit_xgb(xtr, ytr, xva, yva, params, [], es, n_jobs=1, seed=seed)
    return j, h, m.predict(xte), m.best_iteration


def train_xgb_per_segment(
    p: Prepared, cfg: dict, tabs: dict[str, Tabular], log=print
) -> tuple[np.ndarray, dict]:
    """One model per sensor per horizon, parallelised over sensors with joblib."""
    mc = cfg["model"]
    f, n = cfg["horizon_steps"], p.x_in.shape[1]
    s = len(tabs["test"].anchors)
    pred = np.full((s, f, n), np.nan, dtype=np.float32)
    iters: dict[tuple[int, int], int] = {}
    for h in _horizons(mc["horizons"], f):
        t0 = time.time()
        ytr, mtr = targets(p, "train", h)
        yva, mva = targets(p, "val", h)
        jobs = []
        for j in range(n):
            a, b = mtr[:, j], mva[:, j]
            jobs.append(
                delayed(_fit_segment)(
                    j,
                    h,
                    tabs["train"].X[a, j],
                    ytr[a, j],
                    tabs["val"].X[b, j],
                    yva[b, j],
                    tabs["test"].X[:, j],
                    mc["params"],
                    mc["early_stopping_rounds"],
                    cfg["seed"],
                )
            )
        for j, hh, yp, bi in Parallel(n_jobs=N_JOBS)(jobs):
            pred[:, hh - 1, j] = yp
            iters[(j, hh)] = bi
        log(f"  h={h:2d} {n} sensor models, {time.time() - t0:.0f}s")
    return pred, iters


def train_rf(
    p: Prepared, cfg: dict, tabs: dict[str, Tabular], log=print
) -> tuple[np.ndarray, RandomForestRegressor, dict]:
    """Pooled multi-output Random Forest predicting all horizons jointly.

    Rows with any missing target among the F horizons are dropped (logged).
    """
    mc = cfg["model"]
    f, n = cfg["horizon_steps"], p.x_in.shape[1]
    ys, ms = zip(*[targets(p, "train", h) for h in range(1, f + 1)], strict=True)
    y = np.stack(ys, -1).reshape(-1, f)
    m = np.stack(ms, -1).reshape(-1, f)
    full = m.all(1)
    idx = subsample_rows(full, None, mc["row_subsample"], cfg["seed"])
    info = {
        "train_rows_all_targets_observed": int(full.sum()),
        "train_rows_used": int(len(idx)),
        "share_dropped_missing_target": float(1 - full.mean()),
    }
    log(f"  RF rows: {info}")
    rf = RandomForestRegressor(n_jobs=N_JOBS, random_state=cfg["seed"], **mc["params"])
    t0 = time.time()
    rf.fit(tabs["train"].flat()[idx], y[idx])
    log(f"  fit {time.time() - t0:.0f}s")
    s = len(tabs["test"].anchors)
    pred = rf.predict(tabs["test"].flat()).reshape(s, n, f).transpose(0, 2, 1).astype(np.float32)
    return pred, rf, info
