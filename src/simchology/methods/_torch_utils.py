"""Shared helpers for the neural methods: devices, model loading, dtypes, batching.

Nothing in this module imports torch or transformers at import time.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

from .._optional import require
from ..exceptions import ModelNotAvailableError

T = TypeVar("T")


def resolve_torch_device(device: str) -> str:
    """Resolve ``"auto"`` to the best available torch device (cuda > mps > xpu > cpu)."""
    from ..hardware.device import resolve_device

    return resolve_device(device or "auto")


def import_torch(feature: str):
    return require("torch", feature=feature, extra="embeddings")


def import_transformers(feature: str):
    require("torch", feature=feature, extra="embeddings")
    return require("transformers", feature=feature, extra="embeddings")


def dtype_kwargs(dtype: Any) -> dict:
    """``{"dtype": ...}`` for transformers >= 4.56, ``{"torch_dtype": ...}`` before."""
    if dtype is None:
        return {}
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__.split("+")[0]) >= Version("4.56") else "torch_dtype"
    return {key: dtype}


def seed_torch(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(loader: Callable[[], T], model_id: str, what: str = "model") -> T:
    """Call ``loader`` and turn download/lookup failures into a friendly error."""
    try:
        return loader()
    except (OSError, ValueError) as exc:  # missing files, unknown repo, offline hub, ...
        offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        hint = (
            "Hugging Face downloads are disabled (HF_HUB_OFFLINE=1); download the model while "
            "online, e.g. with simchology.download_resources(), or pass a local folder."
            if offline else
            "Check the model name and your internet connection, or pass the path of a local "
            "model folder. To download models in advance use simchology.download_resources()."
        )
        raise ModelNotAvailableError(f"Could not load the {what} {model_id!r}: {first}\n{hint}") from exc


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield items[start:start + size]


def auto_batch_size(device: str, gpu: int = 32, cpu: int = 16) -> int:
    return gpu if device.split(":")[0] in ("cuda", "mps", "xpu") else cpu
