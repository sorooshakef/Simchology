"""How much memory a device can use right now."""

from __future__ import annotations

import psutil

from .device import device_info


def available_memory(device: str = "auto") -> int:
    """Bytes currently available on ``device`` (system RAM for the CPU)."""
    info = device_info(device)
    if info.available_memory is not None:
        return int(info.available_memory)
    return int(psutil.virtual_memory().available)


def total_memory(device: str = "auto") -> int:
    info = device_info(device)
    if info.total_memory is not None:
        return int(info.total_memory)
    return int(psutil.virtual_memory().total)
