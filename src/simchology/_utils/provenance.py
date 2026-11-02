"""Collect version and environment information so results can be reported and reproduced."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
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


# -- model records -----------------------------------------------------------------------
# Every model a method uses is recorded as {"name", "source", "version", ...size}: the size
# keys ("parameters", "vectors", ...) are only present when they are known.

def hf_model_record(model_id: str, revision: str | None = None) -> dict[str, Any]:
    """A Hugging Face model: the commit of the cached snapshot and the number of parameters."""
    from ..hardware.models import cached_revision, local_parameter_count

    record: dict[str, Any] = {
        "name": model_id, "source": "local" if Path(model_id).is_dir() else "huggingface",
        "version": cached_revision(model_id, revision) or revision,
    }
    n = local_parameter_count(model_id, revision=revision)
    if n:
        record["parameters"] = n
    return record


def spacy_model_record(nlp, name: str | None = None) -> dict[str, Any]:
    """A spaCy pipeline: its package version and the size of its word-vector table."""
    meta = getattr(nlp, "meta", None) or {}
    if name is None:
        name = f"{meta.get('lang')}_{meta.get('name')}" if meta.get("name") else "spaCy pipeline"
    record: dict[str, Any] = {"name": name, "source": "spacy", "version": meta.get("version")}
    rows, width = nlp.vocab.vectors.shape
    if rows and width:
        record.update(vectors=int(rows), vector_dimensions=int(width))
    return record


def _count(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    return f"{n / 1e6:.0f}M" if n >= 1e6 else f"{n:,}"


def format_model(record: Any) -> str:
    """``name (version, size)`` for methods sections (older records are plain strings)."""
    if not isinstance(record, dict):
        return str(record)
    source, version = record.get("source"), record.get("version")
    parts = ["Ollama"] if source == "ollama" else []
    if version and source in ("huggingface", "ollama"):
        parts.append(f"revision {version[:7]}" if source == "huggingface"
                     else f"digest {version.removeprefix('sha256:')[:12]}")
    elif version:
        parts.append(f"{'release' if source == 'systemone' else 'version'} {version}")
    if record.get("run"):
        parts.append(f"run {record['run']}")
    if record.get("parameters"):
        parts.append(f"{_count(record['parameters'])} parameters")
    elif record.get("parameter_size"):
        parts.append(f"{record['parameter_size']} parameters")
    if record.get("quantization"):
        parts.append(str(record["quantization"]))
    if record.get("vectors"):
        parts.append(f"{record['vectors']:,} word vectors")
    name = str(record.get("name")) + (f" @ {record['url']}" if record.get("url") else "")
    return f"{name} ({', '.join(parts)})" if parts else name
