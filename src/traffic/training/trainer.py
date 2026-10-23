"""Train loop with early stopping, LR scheduling, checkpointing and logging.

Training happens in scaled space with a masked loss; validation MAE and all reported
metrics are in mph (inverse-transformed), so early stopping tracks the evaluation metric.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch

from traffic.data.datasets import BatchIterator
from traffic.data.scaling import MaskedStandardScaler
from traffic.data.windows import Windows
from traffic.training.losses import LOSSES


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tcfg: dict,
        scaler: MaskedStandardScaler,
        device: torch.device,
        channels: list[int] | None = None,
        seed: int = 0,
    ):
        self.model, self.t, self.device, self.channels, self.seed = (
            model.to(device),
            tcfg,
            device,
            channels,
            seed,
        )
        self.mean = torch.as_tensor(np.asarray(scaler.mean), dtype=torch.float32, device=device)
        self.std = torch.as_tensor(np.asarray(scaler.std), dtype=torch.float32, device=device)
        self.loss_fn = LOSSES[tcfg.get("loss", "mae")]

    def _scale(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std  # broadcasts over [B, F, N] for scalar or [N] stats

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        ns, p = self.t.get("noise_std", 0.0), self.t.get("sensor_mask_p", 0.0)
        if ns > 0:
            x = x.clone()
            x[..., 0] += torch.randn_like(x[..., 0]) * ns
        if p > 0:  # simulate sensor outages: train-mean speed + is_missing flag
            x = x.clone()
            drop = torch.rand(x.shape[0], 1, x.shape[2], device=x.device) < p
            x[..., 0] = torch.where(drop, torch.zeros_like(x[..., 0]), x[..., 0])
            x[..., 1] = torch.where(drop, torch.ones_like(x[..., 1]), x[..., 1])
        return x

    @torch.no_grad()
    def predict(
        self, w: Windows, batch_size: int = 128, aux: np.ndarray | None = None
    ) -> np.ndarray:
        """[S, F, N] predictions in mph."""
        self.model.eval()
        out = []
        for x, _, _, a in BatchIterator(w, batch_size, False, 0, self.channels, aux):
            y = self._forward(x.to(self.device), a if aux is not None else None)
            out.append((y * self.std + self.mean).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def _forward(self, x: torch.Tensor, aux: torch.Tensor | None) -> torch.Tensor:
        """Plain models take x; hybrids also take the scaled base prediction."""
        if aux is None:
            return self.model(x)
        return self.model(x, self._scale(aux.to(self.device)))

    def val_mae(self, w: Windows, aux: np.ndarray | None = None) -> float:
        pred = self.predict(w, aux=aux)
        y, m = w.targets()
        return float(np.abs(pred - y)[m].mean())

    def fit(
        self,
        train_w: Windows,
        val_w: Windows,
        log=print,
        aux_train: np.ndarray | None = None,
        aux_val: np.ndarray | None = None,
    ) -> pd.DataFrame:
        t = self.t
        opt = torch.optim.Adam(self.model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
        if t.get("scheduler", "plateau") == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["max_epochs"])
        else:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
        it = BatchIterator(train_w, t["batch_size"], True, self.seed, self.channels, aux_train)
        best, best_state, bad, rows = float("inf"), None, 0, []
        for ep in range(1, t["max_epochs"] + 1):
            self.model.train()
            t0, tot, nb = time.time(), 0.0, 0
            for x, y, m, a in it:
                x, y, m = x.to(self.device), y.to(self.device), m.to(self.device)
                x = self._augment(x)
                loss = self.loss_fn(
                    self._forward(x, a if aux_train is not None else None), self._scale(y), m
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), t["clip"])
                opt.step()
                tot += float(loss.detach())
                nb += 1
            v = self.val_mae(val_w, aux_val)
            if t.get("scheduler", "plateau") == "cosine":
                sched.step()
            else:
                sched.step(v)
            rows.append(
                {
                    "epoch": ep,
                    "train_loss_scaled": tot / nb,
                    "val_mae_mph": v,
                    "lr": opt.param_groups[0]["lr"],
                    "seconds": time.time() - t0,
                }
            )
            improved = v < best - 1e-4
            if improved:
                best, bad = v, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
            log(
                f"  ep {ep:3d} train {tot / nb:.4f} val_mae {v:.3f} {'*' if improved else ''} "
                f"({rows[-1]['seconds']:.0f}s)"
            )
            if bad >= t["patience"]:
                log(f"  early stop at epoch {ep}")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return pd.DataFrame(rows)
