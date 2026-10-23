"""Hybrid models: residual (XGB + STGAT residual), gating ensemble, simple average.

Needs: results/hybrid/xgb_base.npz (scripts/make_oof.py) and, for gating/average, STGAT
predictions from scripts/train_dl.py (test in predictions/, val in predictions_val/).

Usage: python scripts/train_hybrid.py --kind residual|gating|average [--seeds 0 1 2] [--debug]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from traffic.config import load_config
from traffic.data.pipeline import prepare
from traffic.data.windows import Windows
from traffic.evaluation.io import load_predictions, save_predictions
from traffic.logging_utils import log_run
from traffic.models.factory import build_model, channel_slice
from traffic.seed import get_device, set_seed
from traffic.training.trainer import Trainer


def _paths(cfg, debug):
    root = Path(cfg["paths"]["results"]) / ("debug" if debug else "")
    return root, root / "predictions", root / "predictions_val"


def run_residual(p, cfg, seeds, debug, base) -> None:
    root, pred_dir, val_dir = _paths(cfg, debug)
    tr_w = p.windows["train"]
    ok = ~np.isnan(base["oof_train"][:, 0, 0])  # block 1 has no OOF forecast -> dropped
    sub = Windows(tr_w.chan, tr_w.y, tr_w.ymask, tr_w.index, tr_w.anchors[ok], tr_w.H, tr_w.F)
    aux_tr = base["oof_train"][ok]
    print(f"residual training windows: {len(sub)} of {len(tr_w)} (OOF block 1 dropped)")
    device, ch = get_device(), channel_slice(cfg)
    if debug:
        cfg["train"]["max_epochs"] = 1
    for seed in seeds:
        set_seed(seed)
        model = build_model(cfg, p.data.adj, p.data.locations, len(p.data.sensor_ids))
        tr = Trainer(model, cfg["train"], p.scaler, device, ch, seed)
        t0 = time.time()
        hist = tr.fit(sub, p.windows["val"], aux_train=aux_tr, aux_val=base["val"])
        secs = time.time() - t0
        for split, aux, d in (("test", base["test"], pred_dir), ("val", base["val"], val_dir)):
            w = p.windows[split]
            y, m = w.targets()
            save_predictions(
                d / f"hybrid_residual_seed{seed}.npz",
                tr.predict(w, aux=aux),
                y,
                m,
                w.target_ts(),
                p.data.sensor_ids,
                {
                    "model": "hybrid_residual",
                    "seed": seed,
                    "split": split,
                    "train_seconds": secs,
                    "epochs": len(hist),
                },
            )
        hist.to_csv(root / "metrics" / "training" / f"hybrid_residual_seed{seed}.csv", index=False)
        log_run(
            f"hybrid_residual_seed{seed}", cfg, {"val_mae": round(float(hist.val_mae_mph.min()), 4)}
        )
        print(f"seed {seed}: best val MAE {hist.val_mae_mph.min():.3f} ({secs:.0f}s)")


def gate_inputs(p, split: str, cfg: dict) -> np.ndarray:
    """[S, F, 16]: horizon one-hot (12), peak flag, tod sin/cos of target, window missing rate."""
    w = p.windows[split]
    ts = pd.DatetimeIndex(w.target_ts().ravel())
    hm = (ts.hour * 60 + ts.minute).to_numpy()
    wk = ts.day_of_week.to_numpy() < 5

    def mins(x):
        return int(x[:2]) * 60 + int(x[3:])

    peak = np.zeros(len(ts), bool)
    for a, b in cfg["peak_windows"].values():
        peak |= (hm >= mins(a)) & (hm < mins(b))
    peak &= wk
    s, f = w.target_ts().shape
    miss = np.array([w.x(i)[..., 1].mean() for i in range(s)], dtype=np.float32)  # is_missing
    feat = np.zeros((s, f, f + 4), np.float32)
    feat[..., :f] = np.eye(f)[None]
    feat[..., f] = peak.reshape(s, f)
    feat[..., f + 1] = np.sin(2 * np.pi * hm / 1440).reshape(s, f)
    feat[..., f + 2] = np.cos(2 * np.pi * hm / 1440).reshape(s, f)
    feat[..., f + 3] = miss[:, None]
    return feat


def _t(a) -> torch.Tensor:
    return torch.as_tensor(a, dtype=torch.float32)


def mix_forecasts(gate, gi, xgb_pred, stgat_pred):
    """y = w*stgat + (1-w)*xgb with w = sigmoid(gate(features)) per (sample, horizon)."""
    w = torch.sigmoid(gate(_t(gi))).squeeze(-1)  # [S, F]
    return w[..., None] * _t(stgat_pred) + (1 - w[..., None]) * _t(xgb_pred), w


def train_gate(gi, xgb_pred, stgat_pred, y, mask):
    """Fit the gate on the first 80% of val; early-stop on the last 20% (ADR-008)."""
    n = len(gi)
    cut = int(0.8 * n)
    gate = torch.nn.Sequential(
        torch.nn.Linear(gi.shape[-1], 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
    )
    opt = torch.optim.Adam(gate.parameters(), lr=1e-2)
    m_all = torch.as_tensor(mask)

    def mae(sl):
        pr, _ = mix_forecasts(gate, gi[sl], xgb_pred[sl], stgat_pred[sl])
        m = m_all[sl]
        return ((pr - _t(y[sl])).abs() * m).sum() / m.sum()

    best, best_state, bad = 1e9, None, 0
    for _ in range(400):
        opt.zero_grad()
        mae(slice(0, cut)).backward()
        opt.step()
        with torch.no_grad():
            v = float(mae(slice(cut, n)))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in gate.state_dict().items()}
        else:
            bad += 1
        if bad > 30:
            break
    gate.load_state_dict(best_state)
    return gate, best


def _load_stgat(root_pred: Path, seed: int):
    return load_predictions(root_pred / f"stgat_seed{seed}.npz")


def run_combo(kind: str, p, cfg, seeds, debug, base) -> None:
    root, pred_dir, val_dir = _paths(cfg, debug)
    xte = base["test"]
    xva = base["val"]
    rows = []
    for seed in seeds:
        ste, sva = _load_stgat(pred_dir, seed), _load_stgat(val_dir, seed)
        if kind == "average":
            pte, pva = 0.5 * xte + 0.5 * ste.y_pred, 0.5 * xva + 0.5 * sva.y_pred
            meta = {}
        else:  # gating: train on the first 80% of val, early-stop on the last 20% (ADR-008)
            set_seed(seed)
            gi_va, gi_te = gate_inputs(p, "val", cfg), gate_inputs(p, "test", cfg)
            gate, best = train_gate(gi_va, xva, sva.y_pred, sva.y_true, sva.mask)
            with torch.no_grad():
                pte, wte = mix_forecasts(gate, gi_te, xte, ste.y_pred)
                pva, _ = mix_forecasts(gate, gi_va, xva, sva.y_pred)
            pte, pva = pte.numpy(), pva.numpy()
            pk = gi_te[..., 12].astype(bool)
            for h in range(12):
                rows.append(
                    (
                        seed,
                        h + 1,
                        float(wte[:, h][pk[:, h]].mean()) if pk[:, h].any() else np.nan,
                        float(wte[:, h][~pk[:, h]].mean()),
                    )
                )
            meta = {"gate_val_holdout_mae": best}
        for split, pr, ref, d in (("test", pte, ste, pred_dir), ("val", pva, sva, val_dir)):
            save_predictions(
                d / f"hybrid_{kind}_seed{seed}.npz",
                pr,
                ref.y_true,
                ref.mask,
                ref.target_ts,
                p.data.sensor_ids,
                {"model": f"hybrid_{kind}", "seed": seed, "split": split, **meta},
            )
    if rows:
        out = root / "metrics" / "gate_weights.csv"
        pd.DataFrame(
            rows, columns=["seed", "horizon", "mean_w_stgat_peak", "mean_w_stgat_offpeak"]
        ).to_csv(out, index=False)
    print(f"saved hybrid_{kind} for seeds {seeds}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["residual", "gating", "average"])
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    cfg = load_config("configs/hybrid/residual_hybrid.yaml")
    torch.set_num_threads(cfg["resources"]["torch_threads"])
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    if args.debug:
        seeds = seeds[:1]
    p = prepare(cfg, cfg["paths"]["raw"], debug=args.debug)
    root = _paths(cfg, args.debug)[0]
    (root / "metrics" / "training").mkdir(parents=True, exist_ok=True)
    base = dict(np.load(root / "hybrid" / "xgb_base.npz"))
    if args.kind == "residual":
        run_residual(p, cfg, seeds, args.debug, base)
    else:
        run_combo(args.kind, p, cfg, seeds, args.debug, base)


if __name__ == "__main__":
    main()
