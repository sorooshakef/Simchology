"""Logging and progress helpers.

Simchology talks to its users (often psychologists working in notebooks) through the
``simchology`` logger. A stderr handler is attached once so that messages are visible
without any logging configuration; ``verbose=False`` arguments silence them.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

_LOGGER_NAME = "simchology"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not getattr(logger, "_simchology_configured", False):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[simchology] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger._simchology_configured = True  # type: ignore[attr-defined]
    return logger


def note(message: str, verbose: bool = True) -> None:
    """Show an informational message to the user unless ``verbose`` is False."""
    if verbose:
        get_logger().info(message)


def progress(
    iterable: Iterable[T], *, total: int | None = None, desc: str = "", enabled: bool = True
) -> Iterator[T]:
    """Wrap ``iterable`` in a tqdm progress bar when ``enabled``."""
    if not enabled:
        yield from iterable
        return
    from tqdm.auto import tqdm

    yield from tqdm(iterable, total=total, desc=desc, leave=False)
