"""Global seeding, determinism flags and device selection."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so runs are repeatable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic(flag: bool = True) -> None:
    """Prefer reproducibility over speed when ``flag`` is true."""
    torch.backends.cudnn.deterministic = flag
    torch.backends.cudnn.benchmark = not flag
    torch.use_deterministic_algorithms(flag, warn_only=True)


def get_device() -> torch.device:
    """cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
