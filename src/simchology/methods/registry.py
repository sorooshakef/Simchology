"""Registry mapping method names (and aliases) to method classes."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import pandas as pd

from .._optional import install_hint, is_available
from ..data.validation import did_you_mean
from ..exceptions import ConfigurationError
from .base import SimilarityMethod

_METHODS: dict[str, type[SimilarityMethod]] = {}
_ALIASES: dict[str, str] = {}
_MODULES = (
    "string", "lexical", "topic", "static_embeddings", "wmd", "contextual", "bertscore",
    "supervised", "prompt", "systemone",
)
_LOADED = False

# Presets for compare_methods()
PRESETS: dict[str, tuple[str, ...]] = {
    "fast": ("jaccard", "levenshtein", "osa", "damerau_levenshtein", "bow", "tfidf", "bm25", "lsa"),
    "standard": ("jaccard", "levenshtein", "osa", "damerau_levenshtein", "bow", "tfidf", "bm25",
                 "lsa", "lda", "word_vectors", "wmd", "sbert", "bertscore"),
    "all": ("jaccard", "levenshtein", "osa", "damerau_levenshtein", "bow", "tfidf", "bm25", "lsa",
            "lda", "word_vectors", "wmd", "sbert", "llm_embedding", "bertscore", "cross_encoder",
            "lora_classifier", "llm", "systemone"),
}


def register_method(cls: type[SimilarityMethod]) -> type[SimilarityMethod]:
    """Class decorator registering a method under its ``info.name`` and aliases."""
    _METHODS[cls.info.name] = cls
    for alias in cls.info.aliases:
        _ALIASES[alias] = cls.info.name
    return cls


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for mod in _MODULES:
        name = f"simchology.methods.{mod}"
        if importlib.util.find_spec(name) is not None:
            importlib.import_module(name)


def _normalize(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def resolve_name(name: str) -> str:
    _ensure_loaded()
    key = _normalize(name)
    key = _ALIASES.get(key, key)
    if key not in _METHODS:
        options = sorted(set(_METHODS) | set(_ALIASES))
        raise ConfigurationError(
            f"Unknown method {name!r}.{did_you_mean(key, options)} "
            "See simchology.list_methods() for all methods."
        )
    return key


def get_method_class(name: str) -> type[SimilarityMethod]:
    return _METHODS[resolve_name(name)]


def make_method(spec: Any, params: dict | None = None) -> SimilarityMethod:
    """Create a method from a name, class or instance (``params`` only for name/class)."""
    params = dict(params or {})
    if isinstance(spec, SimilarityMethod):
        if params:
            raise ConfigurationError("method_params cannot be combined with a method instance.")
        return spec
    cls = spec if isinstance(spec, type) and issubclass(spec, SimilarityMethod) else get_method_class(spec)
    missing = cls.missing_modules()
    if missing:
        from ..exceptions import MissingDependencyError

        hints = sorted({install_hint(m) for m in missing})
        raise MissingDependencyError(
            f"The method '{cls.info.name}' needs {', '.join(missing)}. Install with:\n    "
            + "\n    ".join(hints)
        )
    return cls(**params)


def list_methods(installed_only: bool = False) -> pd.DataFrame:
    """Table of all similarity methods with their properties and installation status."""
    _ensure_loaded()
    rows = []
    for name, cls in _METHODS.items():
        info = cls.info
        missing = [m for m in info.requires_modules if not is_available(m)]
        rows.append({
            "method": name,
            "family": info.family,
            "description": info.description,
            "aliases": ", ".join(info.aliases),
            "symmetric": info.symmetric,
            "needs_coded_data": info.requires_labels,
            "output": info.native_output,
            "cost": info.cost_tier,
            "installed": not missing,
            "install": "" if not missing else "; ".join(sorted({install_hint(m) for m in missing})),
        })
    df = pd.DataFrame(rows)
    order = ["string", "lexical", "topic", "static", "alignment", "contextual", "supervised", "prompt"]
    df["_o"] = df["family"].map({f: i for i, f in enumerate(order)})
    df = df.sort_values(["_o", "method"]).drop(columns="_o").reset_index(drop=True)
    return df[df["installed"]].reset_index(drop=True) if installed_only else df
