"""Collect version and environment information so results can be reported and reproduced."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from .._version import __version__

_TRACKED_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "rapidfuzz", "spacy", "wtpsplit", "torch",
    "transformers", "sentence-transformers", "gensim", "POT", "peft", "accelerate", "ollama",
    "httpx",
)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def collect_provenance(**extra: Any) -> dict[str, Any]:
    return {
        "simchology": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "packages": package_versions(),
        **extra,
    }
