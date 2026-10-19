"""Warnings that point at the user's code rather than at Simchology internals."""

from __future__ import annotations

import os
import sys
import warnings

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def external_stacklevel() -> int:
    """Stack level of the first frame outside the simchology package (for ``warnings.warn``)."""
    frame = sys._getframe(1)
    level = 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_PKG_DIR):
        frame = frame.f_back
        level += 1
    return level


def warn(message: str, category: type[Warning]) -> None:
    warnings.warn(message, category, stacklevel=external_stacklevel() + 1)
