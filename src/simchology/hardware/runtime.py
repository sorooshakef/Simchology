"""Process-level settings that make Simchology behave the same on Windows, macOS and Linux."""

from __future__ import annotations

import os
import sys
import warnings

from ..exceptions import SimchologyWarning

_CONFIGURED = False


def configure_environment() -> None:
    """Set environment variables that must be in place before torch/tokenizers are imported.

    Called once on ``import simchology``. Existing user settings are never overridden.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    # Avoid the fork-after-parallelism warning/deadlock of HF tokenizers in DataLoaders.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if sys.platform == "darwin":
        if "torch" in sys.modules and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
            warnings.warn(
                "torch was imported before simchology, so the CPU fallback for operations "
                "that Apple GPUs (MPS) do not support could not be enabled. If you see "
                "'not implemented for MPS' errors, import simchology before torch or set "
                "PYTORCH_ENABLE_MPS_FALLBACK=1 before starting Python.",
                SimchologyWarning,
                stacklevel=3,
            )
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if sys.platform == "win32":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def training_runtime_kwargs(device: str) -> dict:
    """Keyword arguments for ``transformers.TrainingArguments`` that are safe on ``device``.

    * pinned memory only helps (and is only supported) on CUDA;
    * DataLoader worker processes are disabled because spawning them is fragile on
      Windows/macOS notebooks and gives little speed-up for tokenised in-memory data;
    * mixed precision follows the detected device capabilities (never fp16 training on MPS).
    """
    from .device import device_info

    info = device_info(device)
    kwargs: dict = {
        "dataloader_pin_memory": info.kind == "cuda",
        "dataloader_num_workers": 0,
        "bf16": False,
        "fp16": False,
    }
    if info.kind == "cpu":
        kwargs["use_cpu"] = True
    elif info.bf16:
        kwargs["bf16"] = True
    elif info.kind == "cuda" and info.fp16:
        kwargs["fp16"] = True
    return kwargs
