"""Build a DL model from the config (so ablations are config changes only)."""

from __future__ import annotations

import numpy as np
import torch

from traffic.data.graph import get_adjacency
from traffic.data.windows import CHANNELS
from traffic.models.gru_nograph import GRUNoGraph
from traffic.models.hybrid import ResidualHybrid
from traffic.models.stgat import STGAT


def channel_slice(cfg: dict) -> list[int]:
    """Indices of input channels used (A8 drops the calendar features)."""
    if cfg["model"].get("use_time_features", True):
        return list(range(len(CHANNELS)))
    return [CHANNELS.index("speed"), CHANNELS.index("is_missing")]


def build_model(cfg: dict, w: np.ndarray, locations, n_nodes: int) -> torch.nn.Module:
    mc = {k: v for k, v in cfg["model"].items()}
    horizon = cfg["horizon_steps"]
    c = len(channel_slice(cfg))
    name = mc["name"]
    if name == "gru_nograph":
        return GRUNoGraph(
            n_nodes,
            c,
            horizon,
            mc.get("d", 64),
            mc.get("layers", 2),
            mc.get("dropout", 0.1),
            mc.get("node_embedding", True),
        )
    if name in ("stgat", "hybrid_residual"):
        adj = get_adjacency(
            cfg.get("graph", {}).get("variant", "bidirectional"), w, locations, seed=cfg["seed"]
        )
        net = STGAT(
            torch.as_tensor(adj),
            c,
            horizon,
            mc.get("d", 64),
            mc.get("heads", 4),
            mc.get("layers", 2),
            mc.get("temporal", "tcn"),
            mc.get("dropout", 0.1),
            mc.get("attn_dropout", 0.1),
            mc.get("node_embedding", True),
            mc.get("attention", "gat"),
            mc.get("edge_bias", False),
            mc.get("readout", "last"),
            mc.get("dropedge", 0.0),
            horizon if name == "hybrid_residual" else 0,
        )
        return ResidualHybrid(net) if name == "hybrid_residual" else net
    raise ValueError(name)
