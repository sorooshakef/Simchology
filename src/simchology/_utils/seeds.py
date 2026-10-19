"""Seeding helpers so every random step in Simchology is reproducible."""

from __future__ import annotations

import os
import random
import sys

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and, if already imported, torch (CPU, CUDA and MPS)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        mps = getattr(torch, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            try:
                mps.manual_seed(seed)
            except Exception:  # older torch versions
                pass


def derive_seed(seed: int, offset: int) -> int:
    """Deterministically derive a different seed for a sub-step (split, augmentation, ...)."""
    return int(seed) + int(offset)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
