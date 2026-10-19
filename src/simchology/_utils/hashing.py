"""Stable hashes for cache keys and provenance records."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return repr(obj)


def stable_hash(obj: Any, length: int = 16) -> str:
    """Return a hex digest of ``obj`` that is identical across runs and machines."""
    payload = json.dumps(obj, sort_keys=True, default=_default, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
