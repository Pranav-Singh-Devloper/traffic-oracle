"""YAML config loading with ``inherits`` merging, CLI overrides and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in new.items():
        out[k] = (
            _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def _load(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("inherits", None)
    if parent:
        cfg = _deep_merge(
            _load(
                path.parent / parent
                if (path.parent / parent).exists()
                else path.parents[1] / parent
            ),
            cfg,
        )
    return cfg


def _set_dotted(cfg: dict, key: str, value: str) -> None:
    *parents, last = key.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[last] = yaml.safe_load(value)


def validate(cfg: dict[str, Any]) -> None:
    """Fail loudly on configs that would silently break the protocol."""
    s = cfg["split"]
    if abs(s["train"] + s["val"] + s["test"] - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")
    if cfg["horizon_steps"] < max(cfg["report_horizons"]):
        raise ValueError("horizon_steps must cover all report_horizons")
    if cfg["split_protocol"] not in {"strict", "dcrnn"}:
        raise ValueError("split_protocol must be 'strict' or 'dcrnn'")
    if cfg["input_imputation"] not in {"ffill_tod_mean", "zero_mask"}:
        raise ValueError("input_imputation must be 'ffill_tod_mean' or 'zero_mask'")


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, merge ``inherits``, apply ``key.sub=value`` overrides, validate."""
    cfg = _load(Path(path))
    for item in overrides or []:
        key, value = item.split("=", 1)
        _set_dotted(cfg, key, value)
    validate(cfg)
    return cfg
